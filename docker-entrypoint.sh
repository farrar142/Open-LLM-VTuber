#!/bin/bash

# 가상환경 활성화
source /app/.venv/bin/activate

# 가상환경이 제대로 활성화되었는지 확인
echo "✅ Virtual environment activated: $(which python)"
echo "🐍 Python version: $(python --version)"
echo "📦 Pip location: $(which pip)"

# uv 환경 변수 설정
export UV_PROJECT_ENVIRONMENT=/app/.venv
export VIRTUAL_ENV=/app/.venv

# 전달된 명령어 실행
exec "$@"