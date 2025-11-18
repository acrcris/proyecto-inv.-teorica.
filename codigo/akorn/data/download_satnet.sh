#!/usr/bin/env bash
set -euo pipefail

# Copied and adapted from https://github.com/yilundu/ired_code_release/blob/main/data/download-satnet.sh
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

echo "Downloading SATNet Sudoku dataset..."
fetch "https://powei.tw/sudoku.zip" sudoku.zip
unzip -qq -o sudoku.zip
rm -f sudoku.zip

echo "Downloading SATNet Parity dataset..."
fetch "https://powei.tw/parity.zip" parity.zip
unzip -qq -o parity.zip
rm -f parity.zip
echo "Done."
