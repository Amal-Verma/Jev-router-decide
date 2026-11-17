import unittest
from unittest.mock import patch

from decide.reputation import ReputationLedger, UserProfile, TIERS


class MockModel:
    def __init__(self, artifacts_dir=None):
        self.metadata = {"architecture": "test-reputation", "model_name": "ensemble"}

    def route(self, state):
        probs = {"billing": 0.85, "technical": 0.05, "account": 0.05, "shipping": 0.03, "general": 0.02}
        return "billing", probs, 0.85, False

    def approve(self, state):
        return True, 0.92, False

    def grounded(self, state):
        # By default return high grounding
        return True, 0.96, False


class ReputationUnitTests(unittest.TestCase):
    def setUp(self):
        self.ledger = ReputationLedger(eta_reward=2.0, eta_penalty=5.0)

    def test_initial_state(self):
        user = self.ledger.get_or_create("user_alpha")
        self.assertEqual(user.reputation, 50.0)
        self.assertEqual(user.tier, "Probationary")
        self.assertEqual(user.truthful_count, 0)
        self.assertEqual(user.violation_count, 0)

    def test_truthful_grounded_rewards_reputation(self):
        state = {"draft_reply": "Valid reply", "kb_article": "Valid article"}
        output = {"p_true": 0.95, "abstain": False, "value": True}
        result = self.ledger.record_decision("user_alpha", "grounded", output, state)

        self.assertGreater(result["signal"], 0.0)
        self.assertGreater(result["reputation_delta"], 0.0)
        self.assertGreater(result["current_reputation"], 50.0)
        self.assertEqual(result["truthful_count"], 1)

    def test_hallucinated_grounded_penalizes_reputation(self):
        state = {"draft_reply": "Hallucinated reply", "kb_article": "Contradictory article"}
        output = {"p_true": 0.05, "abstain": False, "value": False}
        result = self.ledger.record_decision("user_beta", "grounded", output, state)

        self.assertLess(result["signal"], 0.0)
        self.assertLess(result["reputation_delta"], 0.0)
        self.assertLess(result["current_reputation"], 50.0)
        self.assertEqual(result["violation_count"], 1)

    def test_abstention_is_neutral(self):
        state = {"draft_reply": "Uncertain", "kb_article": "Short"}
        output = {"p_true": 0.50, "abstain": True, "value": False}
        result = self.ledger.record_decision("user_gamma", "grounded", output, state)

        self.assertEqual(result["signal"], 0.0)
        self.assertEqual(result["reputation_delta"], 0.0)
        self.assertEqual(result["current_reputation"], 50.0)
        self.assertEqual(result["evaluation"], "ABSTAIN_NEUTRAL")

    def test_repeated_violations_trigger_quarantine_and_suspension(self):
        user_id = "bad_actor"
        state = {"amount": 500.0}
        output = {"p_true": 0.05, "abstain": False, "value": False}

        # Apply several penalties
        for _ in range(5):
            self.ledger.record_decision(user_id, "approve", output, state)

        user = self.ledger.get_or_create(user_id)
        self.assertLess(user.reputation, 40.0)
        self.assertIn(user.tier, ("Quarantine", "Suspended"))


class ReputationApiIntegrationTests(unittest.TestCase):
    def setUp(self):
        try:
            from fastapi.testclient import TestClient
            import app as service
        except ImportError:
            self.skipTest("fastapi or its dependencies are not installed in this environment.")
        self.service = service
        self.patcher = patch.object(service, "MODEL", MockModel())
        self.patcher.start()
        self.client = TestClient(service.app)

    def tearDown(self):
        if hasattr(self, "patcher"):
            self.patcher.stop()

    def test_decide_attaches_integrity_payload(self):
        response = self.client.post("/v1/decide", json={
            "request_id": "test-int-1",
            "decision": "grounded",
            "state": {"draft_reply": "Hi", "kb_article": "Hello"},
        }, headers={"X-User-ID": "agent_007"})

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("integrity", data)
        self.assertEqual(data["integrity"]["evaluation"], "GROUNDED_TRUTHFUL")
        self.assertGreater(data["integrity"]["current_reputation"], 50.0)

    def test_get_user_reputation_endpoint(self):
        # Submit a decision first
        self.client.post("/v1/decide", json={
            "request_id": "test-int-2",
            "decision": "approve",
            "state": {"amount": 25.0},
        }, headers={"X-User-ID": "agent_audit"})

        response = self.client.get("/v1/reputation/agent_audit")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["user_id"], "agent_audit")
        self.assertGreaterEqual(data["total_requests"], 1)

    def test_list_reputation_leaderboard(self):
        response = self.client.get("/v1/reputation")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("users", data)
        self.assertIn("tiers", data)

    def test_suspended_user_blocked_with_403(self):
        # Artificially set user to Suspended
        bad_user = self.service.LEDGER.get_or_create("spammer_99")
        bad_user.reputation = 5.0
        bad_user.tier = "Suspended"

        response = self.client.post("/v1/decide", json={
            "request_id": "test-int-blocked",
            "decision": "route",
            "state": {"subject": "Spam", "body": "Spam"},
        }, headers={"X-User-ID": "spammer_99"})

        self.assertEqual(response.status_code, 403)
        self.assertIn("suspended", response.json()["detail"].lower())


if __name__ == "__main__":
    unittest.main()
