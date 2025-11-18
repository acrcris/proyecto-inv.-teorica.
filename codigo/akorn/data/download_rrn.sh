#!/usr/bin/env bash
set -euo pipefail

# Copied and adapted from https://github.com/yilundu/ired_code_release/blob/main/data/download-rrn.sh
# Uses wget if available, otherwise falls back to curl.

fetch() {
  local url="$1"; shift
  local out="$1"; shift
  if command -v wget >/dev/null 2>&1; then
    wget -cq "$url" -O "$out"
  elif command -v curl >/dev/null 2>&1; then
    curl -L -sS "$url" -o "$out"
  else
    echo "Error: neither wget nor curl is installed." >&2
    exit 1
  fi
}

ZIP_FILE="sudoku-hard.zip"
URL="https://www.dropbox.com/s/rp3hbjs91xiqdgc/sudoku-hard.zip?dl=1"

echo "Downloading RRN Sudoku dataset..."
fetch "$URL" "$ZIP_FILE"
unzip -qq -o "$ZIP_FILE"
rm -f "$ZIP_FILE"
rm -rf __MACOSX || true

# Ensure the extracted folder is named sudoku-rrn
if [ -d "sudoku-hard" ]; then
  rm -rf sudoku-rrn || true
  mv sudoku-hard sudoku-rrn
fi

echo "Done."
