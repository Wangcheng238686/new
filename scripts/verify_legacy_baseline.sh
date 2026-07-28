#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}/legacy_baseline"

sha256sum -c SOURCE_SHA256SUMS
echo "Legacy baseline source integrity: OK"
echo "Source commit: eee751738d0c4e9f5fa7fdc017cec56bafe11584"

