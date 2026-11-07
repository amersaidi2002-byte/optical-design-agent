#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m optical_agent.cli generate
python -m optical_agent.cli train
python -m optical_agent.cli benchmark
python -m optical_agent.cli report
python -m optical_agent.cli design
