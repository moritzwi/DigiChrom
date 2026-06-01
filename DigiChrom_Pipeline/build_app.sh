#!/usr/bin/env bash
# Build DigiChrom.app with PyInstaller.
# Run from the DigiChrom_Pipeline directory:  bash build_app.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Install PyInstaller if missing
if ! python -c "import PyInstaller" 2>/dev/null; then
    echo "Installing PyInstaller..."
    pip install pyinstaller
fi

# Clean previous build
rm -rf build dist

echo "Building DigiChrom.app ..."
pyinstaller DigiChrom.spec --noconfirm

echo ""
echo "Done!  →  dist/DigiChrom.app"
echo "To test: open dist/DigiChrom.app"
