#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/FilipHenriksson/cs244_march2.git"
APP_DIR="$HOME/cs244_api"

if [ -d "$APP_DIR/.git" ]; then
    cd "$APP_DIR" && git pull origin main
else
    git clone --no-checkout --filter=blob:none "$REPO_URL" "$APP_DIR"
    cd "$APP_DIR"
    git sparse-checkout init --cone
    git sparse-checkout set \
        api.py \
        requirements.txt \
        sim/__init__.py \
        sim/rate_limiter.py \
        sim/cost_tracker.py \
        sim/trace.py \
        sim/metrics.py \
        schedulers/__init__.py \
        schedulers/backoff.py \
        schedulers/fifo.py \
        schedulers/mapreduce.py \
        schedulers/mapreduce_skip.py \
        schedulers/mapreduce_skip_adaptive.py
    git checkout main
fi

python3 -m venv "$APP_DIR/venv" 2>/dev/null || true
source "$APP_DIR/venv/bin/activate"
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "Done. Start with:"
echo "  cd $APP_DIR && source venv/bin/activate && uvicorn api:app --host 0.0.0.0 --port 8000"
