"""Serving wrapper for the classical baseline (artifacts/baseline.pkl)."""
import os
import pickle

import numpy as np
from scipy.sparse import csr_matrix, hstack

from .tabular import approve_features, grounded_features


class BaselineBackend:
    def __init__(self, artifacts_dir):
        from train_baseline import diff_text, ticket  # noqa: F401  (pickle needs the module)
        self._ticket, self._diff = ticket, diff_text
        with open(os.path.join(artifacts_dir, "baseline.pkl"), "rb") as fh:
            self.m = pickle.load(fh)

    def route_probs(self, s):
        return self.m["route"].predict_proba(self.m["route_vec"].transform([self._ticket(s)]))[0]

    def approve_prob(self, s):
        X = hstack([self.m["route_vec"].transform([self._ticket(s)]),
                    csr_matrix(approve_features(s)[None])]).tocsr()
        return self.m["approve"].predict_proba(X)[0, 1]

    def grounded_prob(self, s):
        X = hstack([self.m["g_vec"].transform([self._diff(s)]),
                    csr_matrix(grounded_features(s)[None])]).tocsr()
        return self.m["grounded"].predict_proba(X)[0, 1]
