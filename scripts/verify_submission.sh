#!/usr/bin/env bash
# Prove the single start command works from a clean environment.
#
# 1. creates a fresh venv in a temp dir and installs requirements.txt only
# 2. starts the service with the README command on a free port
# 3. waits for /health, runs selfcheck.py on the full dev split
# 4. prints schema, latency and total, then stops the server
#
# Usage: scripts/verify_submission.sh [--clone] [--keep]
#   --clone  test a fresh `git clone` of HEAD instead of the working tree
#            (proves the committed files are enough; data/dev.jsonl is copied in)
#   --keep   keep the temp dir (venv, logs, results) for inspection
# Env: PYTHON (interpreter for the venv, default python3),
#      ARTIFACTS_DIR (passed through to the service if set).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
CLONE=0
KEEP=0
for a in "$@"; do
  case "$a" in
    --clone) CLONE=1 ;;
    --keep) KEEP=1 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

TMP="$(mktemp -d -t decide-verify.XXXXXX)"
SERVER_PID=""
cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  if [ "$KEEP" = 1 ]; then echo "kept: $TMP"; else rm -rf "$TMP"; fi
}
trap cleanup EXIT

if [ ! -f "$REPO/data/dev.jsonl" ]; then
  echo "FAIL: $REPO/data/dev.jsonl is missing" >&2; exit 1
fi

if [ "$CLONE" = 1 ]; then
  echo "== cloning HEAD into $TMP/src"
  git clone -q "$REPO" "$TMP/src"
  mkdir -p "$TMP/src/data"
  cp "$REPO/data/dev.jsonl" "$TMP/src/data/dev.jsonl"
  SRC="$TMP/src"
else
  SRC="$REPO"
fi

echo "== fresh venv ($("$PYTHON" --version 2>&1)) in $TMP/venv"
"$PYTHON" -m venv "$TMP/venv"
"$TMP/venv/bin/pip" install -q --upgrade pip
"$TMP/venv/bin/pip" install -q -r "$SRC/requirements.txt"

PORT="$("$TMP/venv/bin/python" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')"
echo "== starting: uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1"
(
  cd "$SRC"
  PATH="$TMP/venv/bin:$PATH" exec uvicorn app:app --host 0.0.0.0 --port "$PORT" --workers 1
) >"$TMP/server.log" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "FAIL: server exited during startup" >&2; cat "$TMP/server.log" >&2; exit 1
  fi
  sleep 0.5
done
if ! curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null; then
  echo "FAIL: /health not ready after 60 s" >&2; cat "$TMP/server.log" >&2; exit 1
fi
echo "== /health: $(curl -fsS "http://127.0.0.1:$PORT/health")"

echo "== selfcheck on full dev split"
(cd "$SRC" && "$TMP/venv/bin/python" selfcheck.py \
  --url "http://127.0.0.1:$PORT/v1/decide" --eval data/dev.jsonl --json "$TMP/results.json") \
  >"$TMP/selfcheck.log" 2>&1 || { echo "FAIL: selfcheck errored" >&2; tail -30 "$TMP/selfcheck.log" >&2; exit 1; }

"$TMP/venv/bin/python" - "$TMP/results.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
s, l = r["schema"], r["latency_ms"]
p95 = l["p95"]
print(f"schema:  {s['valid']}/{s['total']} valid")
print("p95 ms:  " + ", ".join(f"{k} {v:.1f}" for k, v in p95.items()) + f"  (mean {l['mean_p95']:.1f})")
print(f"gates:   {r['gates']}")
print(f"TOTAL:   {r['score']:.2f}")
ok = all(r["gates"].values()) and s["valid"] == s["total"]
print("RESULT:  " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
EOF
