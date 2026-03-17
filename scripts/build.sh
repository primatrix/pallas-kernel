#!/usr/bin/env bash
# build.sh — Build distributable packages for tops.
#
# Usage:
#   ./scripts/build.sh          Build sdist and wheel into dist/
#   ./scripts/build.sh clean    Remove previous build artifacts
#
# The resulting packages in dist/ can be installed via:
#   pip install dist/tops-*.whl
#
# Or install directly from a git remote:
#   pip install git+https://github.com/primatrix/pallas-kernel.git

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [ "${1:-}" = "clean" ]; then
    echo "Cleaning build artifacts..."
    rm -rf dist/ build/ *.egg-info tops.egg-info
    echo "Done."
    exit 0
fi

# Prefer uv for building if available, otherwise fall back to pip/build.
if command -v uv &> /dev/null; then
    echo "Building with uv..."
    uv build
elif command -v python3 &> /dev/null; then
    if ! python3 -m build --help &> /dev/null; then
        echo "Installing build tool..."
        python3 -m pip install --quiet build
    fi
    echo "Building with python -m build..."
    python3 -m build
else
    echo "Error: neither uv nor python3 found on PATH." >&2
    exit 1
fi

echo ""
echo "Build complete. Artifacts:"
ls -lh dist/
