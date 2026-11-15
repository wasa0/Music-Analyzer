#!/usr/bin/env bash
# 実行ファイル版 (Python 不要のフォルダ) を作る: dist/<variant>/MUSICA/
#   ./build_exe.sh                  lite (Demucs なし, 小さい)
#   ./build_exe.sh --variant cpu    Demucs + CPU 版 PyTorch
#   ./build_exe.sh --variant cu130  Demucs + CUDA 13 版 PyTorch (非常に大きい)
#   --zip を付けると配布用 zip も作る
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/tools/get_uv.sh"
export UV_PYTHON_INSTALL_DIR="$ROOT/tools/python"
export MUSICA_UV="$UV"
"$UV" run --no-project --managed-python --python 3.12 "$ROOT/tools/build_exe.py" "$@"
