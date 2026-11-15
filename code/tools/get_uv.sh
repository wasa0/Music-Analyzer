# shellcheck shell=bash
# tools/uv/uv が無ければ uv をダウンロードしてチェックサムを確認する (setup.sh / build_exe.sh から source する)。
# 使い方:  source "$ROOT/tools/get_uv.sh"   → 変数 UV に uv のパスが入る
UV_VERSION="0.12.21"
case "$(uname -s)-$(uname -m)" in
  Linux-x86_64)               _T="x86_64-unknown-linux-gnu";  _SHA="23f02075b652bb1df64178cfae41b5caf160822e720e2663568f3f5d63bc52c0" ;;
  Linux-aarch64|Linux-arm64)  _T="aarch64-unknown-linux-gnu"; _SHA="030b69227b40af8c1981b7301793dc66e71ed3c796ea8688209dd268bd91ec51" ;;
  Darwin-arm64)               _T="aarch64-apple-darwin";      _SHA="b88bda573e566ef9bced66b155fe0408626fbbc053aee1c30ba686f0728c9447" ;;
  Darwin-x86_64)              _T="x86_64-apple-darwin";       _SHA="2b336763b396ec6afa20c5a8b083538ca7402445b868311979d740a4344c17d8" ;;
  *) echo "この環境 ($(uname -s) $(uname -m)) には対応していません"; exit 1 ;;
esac

_UV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/uv"
UV="$_UV_DIR/uv"
if [ ! -x "$UV" ] || [ "$("$UV" --version 2>/dev/null | awk '{print $2}')" != "$UV_VERSION" ]; then
  echo "uv $UV_VERSION をダウンロードしています..."
  _tmp="$(mktemp -d)"
  curl -fsSL -o "$_tmp/uv.tar.gz" "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-$_T.tar.gz"
  if command -v sha256sum >/dev/null 2>&1; then
    _got="$(sha256sum "$_tmp/uv.tar.gz" | awk '{print $1}')"
  else
    _got="$(shasum -a 256 "$_tmp/uv.tar.gz" | awk '{print $1}')"
  fi
  if [ "$_got" != "$_SHA" ]; then
    rm -rf "$_tmp"
    echo "ダウンロードした uv のチェックサムが一致しません (中断します)"; exit 1
  fi
  tar -xzf "$_tmp/uv.tar.gz" -C "$_tmp"
  mkdir -p "$_UV_DIR"
  cp "$_tmp/uv-$_T/uv" "$_tmp/uv-$_T/uvx" "$_UV_DIR/"
  chmod +x "$_UV_DIR/uv" "$_UV_DIR/uvx"
  rm -rf "$_tmp"
fi
unset _T _SHA _UV_DIR _tmp _got
