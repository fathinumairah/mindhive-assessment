#!/bin/bash
export PATH="/opt/render/project/src/.venv/bin:$PATH"
python -m uvicorn main:app --host 0.0.0.0 --port $PORT 