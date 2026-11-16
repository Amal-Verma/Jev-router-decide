import importlib.util
import json
import math
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from decide.runtime import EncoderBackend
from model import DecisionModel, QUEUES

ROOT = Path(__file__).resolve().parent


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.model = DecisionModel.__new__(DecisionModel)
        self.model.policy = json.loads((ROOT / "artifacts/policy.json").read_text())
        self.model.backend = Mock()
        self.model.backend.route_probs.return_value = [0.92, 0.02, 0.02, 0.02, 0.02]
        self.model.backend.approve_prob.return_value = 0.9
        self.model.backend.grounded_prob.return_value = 0.9999
        self.state = {"subject": "Refund", "body": "Please refund a duplicate charge.", "amount": 24, "customer_tier": "pro", "draft_reply": "Refunds take 5 days.", "kb_article": "Refunds take 5 days."}

    def test_missing_evidence_abstains_without_calling_encoder(self):
        for method in (self.model.route, self.model.approve, self.model.grounded):
            self.assertTrue(method({})[-1])
        self.assertEqual(list(self.model.route({})[1].values()), [0.2] * 5)
        self.assertEqual(self.model.approve({}), (False, 0.5, True))
        self.assertEqual(self.model.grounded(dict(self.state, kb_article="  ")), (False, 0.5, True))
        self.model.backend.assert_not_called()
        self.assertEqual(self.model.backend.mock_calls, [])

    def test_approval_preserves_cost_threshold_and_asymmetric_band(self):
        for probability, value, abstain in [(0.15, False, False), (0.2, False, True), (0.75, False, True), (0.76, True, True), (0.79, True, False), (0.9, True, False)]:
            with self.subTest(probability=probability):
                self.model.backend.approve_prob.return_value = probability
                self.assertEqual(self.model.approve(self.state), (value, probability, abstain))

    def test_negative_amount_defers_instead_of_log_domain_error(self):
        self.assertEqual(self.model.approve(dict(self.state, amount=-24)), (False, 0.5, True))
        self.model.backend.approve_prob.assert_not_called()

    def test_routing_confidence_matches_returned_queue(self):
        value, probabilities, confidence, abstain = self.model.route(self.state)
        self.assertEqual(value, "billing")
        self.assertEqual(set(probabilities), set(QUEUES))
        self.assertAlmostEqual(sum(probabilities.values()), 1)
        self.assertEqual(confidence, probabilities[value])
        self.assertFalse(abstain)

    def test_sigmoid_handles_extreme_logits(self):
        backend = EncoderBackend.__new__(EncoderBackend)
        backend.temp = {"approve": 1}
        backend.maps = {}
        for logit, expected in [(-10000, 0.0), (10000, 1.0), (0, 0.5)]:
            with patch.object(backend, "_logits", return_value=logit):
                self.assertEqual(backend._binary("approve", {}), expected)


class RealModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DecisionModel(str(ROOT / "artifacts"))

    def test_loaded_graph_uses_only_cpu(self):
        self.assertEqual(self.model.backend.sess.get_providers(), ["CPUExecutionProvider"])
        self.assertEqual(self.model.metadata["model_name"], "ensemble")

    def test_complete_records_match_original_policy(self):
        source = subprocess.check_output(["git", "show", "1559b331a2cffb5b94adf6484daa1b093d38dbfb:model.py"], cwd=ROOT, text=True)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "original_model.py"
            path.write_text(source)
            spec = importlib.util.spec_from_file_location("original_decide_model", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        original = module.DecisionModel.__new__(module.DecisionModel)
        original.backend = self.model.backend
        original.policy = self.model.policy
        rows = [
            {"subject": "Invoice request", "body": "Please send my invoice.", "customer_tier": "enterprise", "amount": 50, "draft_reply": "Refunds take 5 days.", "kb_article": "Refunds take 5 days."},
            {"subject": "Tracking not updated", "body": "My order has not arrived.", "customer_tier": "free", "amount": 0, "draft_reply": "Delivery takes 1 day.", "kb_article": "Delivery takes 7 days."},
        ]
        for state in rows:
            for decision in ("route", "approve", "grounded"):
                with self.subTest(decision=decision, subject=state["subject"]):
                    before = getattr(original, decision)(state)
                    after = getattr(self.model, decision)(state)
                    self.assertEqual(before, after)

    def test_inference_uses_no_network(self):
        state = {"subject": "Refund please", "body": "Duplicate payment.", "customer_tier": "pro", "amount": 24, "draft_reply": "Refunds take 5 days.", "kb_article": "Refunds take 5 days."}
        with patch("socket.socket", side_effect=AssertionError("Network access during inference")):
            route = self.model.route(state)
            approval = self.model.approve(state)
            grounding = self.model.grounded(state)
        self.assertTrue(all(math.isfinite(value) for value in route[1].values()))
        self.assertTrue(all(0 <= value <= 1 for value in [approval[1], grounding[1]]))


if __name__ == "__main__":
    unittest.main()
