#!/usr/bin/env python3
"""オーディオビジュアライザー単体 (GUI). 解析 (音源分離) をしなくても、ファイル / YouTube をすぐ可視化できる.

FFT スペクトラム・倍音分解 (正弦波の足し算)・位相トリガー波形・ハーモニック・マンダラ・
スクロールするスペクトログラム・コード (五度圏) をリアルタイム表示する。
解析結果フォルダを開けば、ボーカル / フロント楽器 / カラオケ / 全パート を選んで (または並べて) 表示できる。
歌声アナライザーとタブで切り替えられる統合版は musica_ex.py。

使い方:
    python audio_visualizer.py                         # 起動してファイル / YouTube を選ぶ
    python audio_visualizer.py --file song.mp3
    python audio_visualizer.py --url "https://www.youtube.com/watch?v=XXXX" --start 60
    python audio_visualizer.py --result cache/results/<フォルダ> [--source vocal]
"""
import sys

from musica.app import main

if __name__ == "__main__":
    sys.exit(main(default_mode="visualizer"))
