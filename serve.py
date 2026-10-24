"""Start the Decide API with a chosen model.

    python serve.py                       # ensemble (same as the judges' `uvicorn app:app`)
    python serve.py --model minilm        # MiniLM + MoE alone
    python serve.py --model ettin         # Ettin-17m + MoE alone
    python serve.py --model ettin --port 8001 --host 127.0.0.1

All three expose the same POST /v1/decide endpoint and response schema.
"""
import argparse
import os

import uvicorn

MODELS = {"ensemble": "artifacts", "minilm": "artifacts/minilm", "ettin": "artifacts/ettin"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=MODELS, default="ensemble")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    os.environ["ARTIFACTS_DIR"] = os.path.join(here, MODELS[args.model])
    print(f"serving model '{args.model}' from {MODELS[args.model]} on {args.host}:{args.port}", flush=True)
    uvicorn.run("app:app", host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
