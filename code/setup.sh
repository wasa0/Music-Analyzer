#!/usr/bin/env bash
# MUSICA_EX セットアップ (Linux / macOS)
#   ./setup.sh                 自動 (GPU があれば CUDA 版 PyTorch + Demucs)
#   ./setup.sh --gpu cpu       CPU 版 PyTorch
#   ./setup.sh --gpu none      Demucs なし (軽量)
#   ./setup.sh --uninstall     作成したショートカットを削除
# Python の事前インストールは不要 (uv が Python 3.12 を tools/ に用意する)。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/tools/get_uv.sh"

# Python も tools/ に置く (フォルダごと削除すればきれいに消える)
export UV_PYTHON_INSTALL_DIR="$ROOT/tools/python"
export MUSICA_UV="$UV"
"$UV" run --no-project --managed-python --python 3.12 "$ROOT/tools/install.py" "$@"
