import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import app as service
from model import QUEUES
from selfcheck import validate


class ExampleModel:
    def __init__(self, artifacts_dir):
        self.artifacts_dir = artifacts_dir
        self.metadata = {"architecture": "test-shared", "private_path": "/secret/model"}
        self.last_state = None

    def route(self, state):
        self.last_state = state
        probs = dict.fromkeys(QUEUES, 0.05)
        probs["billing"] = 0.80
        return "billing", probs, 0.80, False

    def approve(self, state):
        self.last_state = state
        return False, 0.60, True

    def grounded(self, state):
        self.last_state = state
        return True, 0.90, False


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.model_patch = patch.object(service, "DecisionModel", ExampleModel)
        self.model_patch.start()
        self.client = TestClient(service.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.model_patch.stop()

    def decide(self, decision, state=None):
        return self.client.post("/v1/decide", json={
            "request_id": "contract-check",
            "decision": decision,
            "state": state or {},
        })

    def test_all_decisions_match_judging_schema(self):
        for decision in ("route", "approve", "grounded"):
            with self.subTest(decision=decision):
                response = self.decide(decision)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(validate(response.json(), decision, "contract-check"), [])
        self.assertTrue(self.decide("approve").json()["abstain"])
        self.assertFalse(self.decide("approve").json()["value"])

    def test_nulls_are_normalized_and_extra_fields_preserved(self):
        state = {name: None for name in ("subject", "body", "customer_tier", "amount", "draft_reply", "kb_article")}
        state["account_age"] = 100
        self.assertEqual(self.decide("route", state).status_code, 200)
        self.assertEqual(service.MODEL.last_state, {
            "subject": "", "body": "", "customer_tier": "", "amount": 0.0,
            "draft_reply": "", "kb_article": "", "account_age": 100,
        })

    def test_route_normalizes_distribution_and_matching_confidence(self):
        probabilities = dict.fromkeys(QUEUES, 0.1)
        probabilities["technical"] = 1.6
        with patch.object(service.MODEL, "route", return_value=("technical", probabilities, 0.99, False)):
            response = self.decide("route")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertAlmostEqual(sum(payload["probs"].values()), 1.0)
        self.assertAlmostEqual(payload["confidence"], payload["probs"]["technical"])
        self.assertEqual(validate(payload, "route", "contract-check"), [])

    def test_startup_propagates_missing_artifact_error(self):
        with patch.object(service, "DecisionModel", side_effect=FileNotFoundError("Train the model first.")):
            with self.assertRaisesRegex(FileNotFoundError, "Train the model first"):
                with TestClient(service.app):
                    pass

    def test_invalid_decision_rejected(self):
        self.assertEqual(self.decide("unknown").status_code, 422)

    def test_nonfinite_amounts_rejected_with_json_error(self):
        for amount in ("nan", "inf", "-inf", float("nan"), float("inf")):
            with self.subTest(amount=amount):
                payload = {"request_id": "invalid", "decision": "approve", "state": {"amount": amount}}
                response = self.client.post("/v1/decide", content=json.dumps(payload), headers={"Content-Type": "application/json"})
                self.assertEqual(response.status_code, 422)
                self.assertIn("detail", response.json())

    def test_health_uses_safe_metadata(self):
        health = self.client.get("/health").json()
        self.assertTrue(health["loaded"])
        self.assertEqual(health["metadata"], {"architecture": "test-shared"})
        self.assertTrue(Path(service.MODEL.artifacts_dir).is_absolute())

    def test_metrics_missing_and_available(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.json"
            with patch.object(service, "RESULTS_PATH", path):
                self.assertEqual(self.client.get("/v1/metrics").status_code, 404)
                path.write_text('{"score": 42}', encoding="utf-8")
                response = self.client.get("/v1/metrics")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), {"score": 42})

    def test_frontend_is_served(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])

    def test_other_models_do_not_show_ensemble_metrics(self):
        service.MODEL.metadata["model_name"] = "minilm"
        self.assertEqual(self.client.get("/v1/metrics").status_code, 404)

    def test_static_script_and_offline_fonts_are_served(self):
        for path in ("/static/app.js", "/static/styles.css", "/static/fonts/NotoSans-Regular.ttf"):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 200)

    def test_unloaded_model_returns_service_unavailable(self):
        with patch.object(service, "MODEL", None):
            self.assertEqual(self.decide("route").status_code, 503)

    def test_invalid_model_probabilities_are_not_emitted(self):
        with patch.object(service.MODEL, "grounded", return_value=(True, float("nan"), False)):
            self.assertEqual(self.decide("grounded").status_code, 500)
        with patch.object(service.MODEL, "route", return_value=("billing", dict.fromkeys(QUEUES, 0), 0, False)):
            self.assertEqual(self.decide("route").status_code, 500)


if __name__ == "__main__":
    unittest.main()
