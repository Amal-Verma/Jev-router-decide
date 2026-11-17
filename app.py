import math
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from model import QUEUES, DecisionModel
from decide.reputation import ReputationLedger, UserProfile, TIERS
LEDGER = ReputationLedger()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
RESULTS_PATH = BASE_DIR / "results.json"
PROB_TOLERANCE = 0.01
MODEL: Optional[DecisionModel] = None
PUBLIC_METADATA_KEYS = {
    "architecture", "trained_at", "train_records", "validation_records",
    "calibration_records", "feature_count", "artifact_bytes", "seed",
    "versions", "thresholds", "model_name", "name", "hidden_size",
}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global MODEL
    artifacts_dir = os.environ.get("ARTIFACTS_DIR", str(BASE_DIR / "artifacts"))
    MODEL = DecisionModel(artifacts_dir)
    yield
    MODEL = None


app = FastAPI(title="Decide", docs_url=None, redoc_url=None, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR), check_dir=False), name="static")


class State(BaseModel):
    subject: str = ""
    body: str = ""
    customer_tier: str = "free"
    amount: float = Field(default=0.0, allow_inf_nan=False)
    draft_reply: str = ""
    kb_article: str = ""

    model_config = {"extra": "allow"}

    @field_validator("subject", "body", "customer_tier", "draft_reply", "kb_article", mode="before")
    @classmethod
    def blank_if_null(cls, value: Any) -> Any:
        return "" if value is None else value

    @field_validator("amount", mode="before")
    @classmethod
    def zero_if_null(cls, value: Any) -> Any:
        return 0.0 if value is None else value


class DecideRequest(BaseModel):
    request_id: str
    decision: Literal["route", "approve", "grounded"]
    state: State


@app.exception_handler(RequestValidationError)
async def validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = [
        {"type": error["type"], "loc": error["loc"], "msg": error["msg"]}
        for error in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/")
def interface() -> FileResponse:
    page = STATIC_DIR / "index.html"
    if not page.is_file():
        raise HTTPException(status_code=404, detail="The interface has not been installed.")
    return FileResponse(page)


@app.get("/health")
def health() -> Dict[str, Any]:
    metadata = getattr(MODEL, "metadata", {}) if MODEL is not None else {}
    return {
        "status": "ok" if MODEL is not None else "unavailable",
        "loaded": MODEL is not None,
        "metadata": {key: value for key, value in metadata.items() if key in PUBLIC_METADATA_KEYS},
    }


@app.get("/v1/metrics")
def metrics() -> FileResponse:
    model_name = getattr(MODEL, "metadata", {}).get("model_name", "ensemble")
    report = RESULTS_PATH if model_name == "ensemble" else BASE_DIR / "evaluation" / f"{model_name}.json"
    if model_name not in {"ensemble", "minilm", "ettin"} or not report.is_file():
        raise HTTPException(status_code=404, detail="Run selfcheck.py to generate evaluation results.")
    return FileResponse(report, media_type="application/json")


@app.post("/v1/decide")
def decide(req: DecideRequest, request: Request) -> JSONResponse:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="The model is not loaded.")
    started_at = time.perf_counter()

    user_id = request.headers.get("X-User-ID") or request.query_params.get("user_id")
    if user_id:
        user = LEDGER.get_or_create(user_id)
        if user.tier == "Suspended":
            raise HTTPException(status_code=403, detail="Account suspended due to repeated integrity violations.")
        if user.tier == "Quarantine":
            time.sleep(1.5)

    state = req.state.model_dump()

    if req.decision == "route":
        value, probs, confidence, abstain = MODEL.route(state)
        probs = {queue: float(probs.get(queue, 0.0)) for queue in QUEUES}
        total = sum(probs.values())
        if any(not math.isfinite(probability) or probability < 0 for probability in probs.values()) or not math.isfinite(total) or total <= 0:
            raise HTTPException(status_code=500, detail="The model returned an invalid distribution.")
        if abs(total - 1.0) > PROB_TOLERANCE:
            probs = {queue: probability / total for queue, probability in probs.items()}
        if value not in probs:
            value = max(probs, key=probs.get)
        body: Dict[str, Any] = {
            "type": "choice",
            "value": value,
            "probs": probs,
            "confidence": float(probs[value]),
            "abstain": bool(abstain),
        }
    else:
        predict = MODEL.approve if req.decision == "approve" else MODEL.grounded
        value, p_true, abstain = predict(state)
        p_true = float(p_true)
        if not math.isfinite(p_true) or not 0.0 <= p_true <= 1.0:
            raise HTTPException(status_code=500, detail="The model returned an invalid probability.")
        body = {
            "type": "binary",
            "value": bool(value),
            "p_true": p_true,
            "abstain": bool(abstain),
        }

    body["request_id"] = req.request_id
    body["decision"] = req.decision
    body["latency_ms"] = round((time.perf_counter() - started_at) * 1000.0, 3)

    if user_id:
        integrity_info = LEDGER.record_decision(user_id, req.decision, body, state)
    else:
        signal, evaluation = LEDGER.calculate_signal(req.decision, body, state)
        integrity_info = {
            "signal": round(signal, 3),
            "evaluation": evaluation,
            "reputation_delta": 0.0,
            "current_reputation": 50.0,
            "user_tier": "Guest / Benchmark",
            "truthful_count": 0,
            "violation_count": 0,
        }
    body["integrity"] = integrity_info

    return JSONResponse(body)


@app.get("/v1/reputation")
def list_reputation() -> Dict[str, Any]:
    users_summary = [
        {
            "user_id": u.user_id,
            "reputation": round(u.reputation, 2),
            "tier": u.tier,
            "total_requests": u.total_requests,
            "truthful_count": u.truthful_count,
            "violation_count": u.violation_count,
        }
        for u in sorted(LEDGER.users.values(), key=lambda x: x.reputation, reverse=True)
    ]
    return {
        "total_tracked_users": len(LEDGER.users),
        "users": users_summary,
        "tiers": TIERS,
    }


@app.get("/v1/reputation/{user_id}")
def get_user_reputation(user_id: str) -> Dict[str, Any]:
    user = LEDGER.get_or_create(user_id)
    return {
        "user_id": user.user_id,
        "reputation": round(user.reputation, 2),
        "tier": user.tier,
        "total_requests": user.total_requests,
        "truthful_count": user.truthful_count,
        "violation_count": user.violation_count,
        "quota_remaining": user.quota_remaining,
    }


@app.post("/v1/reputation/reset")
def reset_user_reputation(payload: Dict[str, str]) -> Dict[str, Any]:
    user_id = payload.get("user_id", "default_agent")
    if user_id in LEDGER.users:
        LEDGER.users[user_id] = UserProfile(user_id=user_id)
    return {"status": "reset", "user_id": user_id, "reputation": 50.0, "tier": "Probationary"}
