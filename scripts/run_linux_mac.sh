#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python src/brain_age_multiscale.py
