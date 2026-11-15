"""PyInstaller で作る実行ファイルの入口 (MUSICA.exe)."""
import sys

from musica.app import main

if __name__ == "__main__":
    sys.exit(main())
