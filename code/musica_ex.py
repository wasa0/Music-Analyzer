#!/usr/bin/env python3
"""MUSICA 統合版 (GUI): 歌声アナライザーとオーディオビジュアライザーをタブで切り替える.

「歌声アナライザー」タブで YouTube / ファイルの区間をボーカル・楽器に分離して解析し、
「ビジュアライザー」タブで同じ区間を倍音・スペクトラム・スペクトログラム・コードで可視化する
(ボーカル / フロント楽器 / カラオケ / 全パート を選んで、または並べて表示できる)。

使い方:
    python musica_ex.py
    python musica_ex.py --url "https://www.youtube.com/watch?v=XXXX" --start 60 --end 90
    python musica_ex.py --result cache/results/<フォルダ> [--tab visualizer]

単体版: vocal_analyzer.py (歌声アナライザーのみ) / audio_visualizer.py (解析せずに可視化)
"""
import sys

from musica.app import main

if __name__ == "__main__":
    sys.exit(main(default_mode="integrated"))
