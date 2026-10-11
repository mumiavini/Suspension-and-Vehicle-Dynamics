#!/usr/bin/env sh
# VD Studio launcher (Linux / macOS).
cd "$(dirname "$0")" || exit 1
exec uv run --extra desktop python -m apps.vdstudio "$@"
