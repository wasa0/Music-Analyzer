#!/usr/bin/env python3
"""歌声調教アナライザー単体 (GUI).

YouTube の URL / 検索結果 / ローカルファイルから音源を取り込み、指定区間をボーカル・楽器に分離して
しゃくり・オーバーシュート・ビブラート・語尾のフォール・上ずり等の「調教」特徴量とコード進行を解析する。
ビジュアライザーのタブ付きの統合版は musica_ex.py。

使い方:
    python vocal_analyzer.py
    python vocal_analyzer.py --url "https://www.youtube.com/watch?v=XXXX" --start 60 --end 90
    python vocal_analyzer.py --result cache/results/<フォルダ>
"""
import sys

from musica.app import main

if __name__ == "__main__":
    sys.exit(main(default_mode="analyzer"))
