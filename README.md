# MUSIC analyzar

実際の歌声から「調教」の特徴を計測する**歌声アナライザー**と、倍音・スペクトラム・コードを動きで見せる**オーディオビジュアライザー**です。調教とは、VOCALOID などで手作業で付けるピッチや音量の表情のことです。

YouTube（URL / 検索）や手元の音声・動画ファイルから区間を切り出し、ボーカルと楽器に分離して解析します。Windows / macOS / Linux で動きます。Python を別にインストールする必要はありません（セットアップが用意します）。

アプリ本体はすべて [`code/`](code/) フォルダにあります。以下のコマンドは `code` フォルダで実行してください。

## 3 つの起動方法

| アプリ | ショートカット名 | 使いどころ |
|---|---|---|
| 統合版 | MUSICA | 「歌声アナライザー」と「ビジュアライザー」をタブで切り替える。解析した区間をそのままパート別に可視化できる |
| 歌声アナライザー単体 | MUSICA 歌声アナライザー | 解析だけ使う（ビジュアライザーのタブなし） |
| ビジュアライザー単体 | MUSICA ビジュアライザー | 解析（音源分離）をせずに、ファイルや YouTube をすぐ可視化する |

### 歌声アナライザーでできること

- 音源分離: Demucs（4 パート: ボーカル / ドラム / ベース / その他、6 パート: さらにギター / ピアノ）または軽量な簡易分離
- ピッチ: pYIN で求めた f0 を FFT の倍音で精密化し、カラオケ風の音程バーにする
- ノートごとの調教の特徴量
  - 入り方（しゃくり / 上から入る / ポルタメント / ストレート）、オーバーシュート
  - ビブラート（速さ・深さ・かかり始め）
  - 語尾（フォール / 語尾上げ / 予備動作）
  - 上ずり・こぶし・ひっくり返り、音程のずれ
  - 音量（DYN）、音色（HNR = 息っぽさ、明るさ、H1-H2）
- VOCALOID 風のコントロールカーブ（PIT / DYN / BRE / BRI）
- 伴奏パートから求めたコード進行・キー・テンポと、各ノートとコードの関係（コードトーンかテンションか）
- 歌い方の傾向のまとめ、CSV 書き出し

### ビジュアライザーでできること

- 6 種類のパネルをリアルタイム表示: ハーモニック・マンダラ / 倍音分解（正弦波の足し算） / 基音の位相でトリガーした波形 / FFT スペクトラム / スペクトログラム + ピッチ軌跡 / コード・五度圏
- 解析結果を表示しているときは、ボーカル / フロント楽器 / カラオケ / 全パートを選んで表示したり、並べて比較したりできる（[表示パート](#表示パート)）

## セットアップ

| OS | 手順 |
|---|---|
| Windows | `code\setup_windows.bat` をダブルクリック |
| Linux / macOS | `cd code` → `./setup.sh` |

セットアップは次を自動で行います。初回は数分〜十数分かかります。

1. uv（Python のパッケージ管理ツール。チェックサムを確認して取得）と Python 3.12 を `code/tools/` に用意し、`code/.venv` に仮想環境を作る
2. 版を固定したライブラリとアプリ本体を入れる（ffmpeg と、YouTube の取得に使う deno も含む）
3. GPU を調べて、合った PyTorch と Demucs を入れる
4. Demucs のモデル（htdemucs、約 80MB）を先にダウンロードし、動作チェックをする
5. ショートカットを作る（Windows: デスクトップとスタートメニューの「MUSICA_EX」、Linux: アプリ一覧、macOS: `code` フォルダの `.command`）

GPU の判定で入る PyTorch の種類:

| 環境 | 入る PyTorch |
|---|---|
| NVIDIA、ドライバ 580 以降（RTX 20 シリーズ以降） | CUDA 13.0 版 |
| NVIDIA、ドライバ 528 以降（Linux は 525 以降） | CUDA 12.6 版 |
| NVIDIA のドライバが古い / NVIDIA の GPU なし | CPU 版 |
| Apple Silicon の Mac | 標準版（GPU = MPS） |
| Intel の Mac | Demucs なし（簡易分離のみ） |

RTX 50 シリーズを GPU で使うには、ドライバ 580 以降が必要です。PyTorch の容量は GPU 版で約 7GB、CPU 版で約 2GB です。

オプション（例: `setup_windows.bat --gpu cpu`、`./setup.sh --gpu none`）:

| オプション | 内容 |
|---|---|
| `--gpu auto\|cuda\|cpu\|none` | PyTorch の種類（`none` = Demucs を入れない） |
| `--no-shortcuts` / `--shortcut-dir DIR` | ショートカットを作らない / 作る場所を変える |
| `--no-model` / `--no-selftest` | モデルの事前ダウンロード / 動作チェックをしない |
| `--uninstall` | 作ったショートカットを削除する |

- 何度実行しても大丈夫です。仮想環境は再利用され、ショートカットは作り直されます。
- アップデートでショートカットが増えたとき（統合版の「MUSICA」など）も、セットアップをもう一度実行してください。
- アンインストールは `uninstall_windows.bat`（Linux / macOS は `./setup.sh --uninstall`）でショートカットを消し、フォルダごと削除します。
- 手動で入れる場合は `pip install -r requirements.txt` です（Demucs は [`code/requirements-demucs.txt`](code/requirements-demucs.txt) を参照）。

## 起動

ショートカットから起動するか、`code` フォルダで次のコマンドを実行します（Windows は `.venv/bin/python` を `.venv\Scripts\python` に読み替え）。

```bash
.venv/bin/python -m musica.app                # 統合版
.venv/bin/python -m musica.app --analyzer     # 歌声アナライザー単体
.venv/bin/python -m musica.app --visualizer   # ビジュアライザー単体
```

起動用のスクリプト `musica_ex.py`（統合版）、`vocal_analyzer.py`（アナライザー単体）、`audio_visualizer.py`（ビジュアライザー単体）も同じように使えます。

## 使い方（統合版）

1. 「🎤 歌声アナライザー」タブの「1. 音源を選ぶ」で、音源を次のどれかで取り込む
   - URL を貼る（`t=90` 付きなら開始位置も反映）
   - 「検索」タブで検索して、結果をダブルクリック
   - 「ファイル」タブで音声 / 動画ファイルを開く
2. 波形の青い範囲をドラッグするか、開始・終了を入力して区間を決める（「▶ 区間を試聴」で確認できる）
3. 「2. 解析設定」で音源分離・声域・デバイスを選んで「解析開始」
4. 結果を見る: 「音程バー + コード + ノート詳細」「パート (楽器分離)」「ノート一覧」「歌い方の傾向」「スペクトログラム」の各タブ
5. 「🎆 ビジュアライザーで見る」（または「🎆 ビジュアライザー」タブ）で同じ区間を可視化する。再生はタブを切り替えても続く

- 解析結果は自動で保存され、「ファイル → 解析結果を開く」で後から開けます。
- 同じ区間・同じ分離方式で解析し直すと、分離済みのパートを再利用します。
- 区間が長いほど時間がかかります（240 秒を超えると警告が出ます）。

## 表示パート

ビジュアライザーで解析結果を表示しているとき、「表示:」のボタンで、どのパートの音で描くかを選べます。コード・五度圏はいつも全体を表示します。

| 表示 | 使うパート | ピッチ軌跡 |
|---|---|---|
| ボーカル | ボーカル | 解析結果（音程バー付き） |
| フロント楽器 | ボーカルとリズム隊（ドラム・ベース）以外。4 パート分離なら「その他」、6 パートならギター + ピアノ + その他 | そのパートの音から推定 |
| カラオケ | ボーカル以外のすべて | そのパートの音から推定 |
| 全パート | すべて（元の曲） | 解析結果（ボーカル） |

- 表示と再生は別々に選べます。鳴らすパートは右側の「再生:」のボタンで選びます。
- レイアウトの「パート別に並べる: …」を選ぶと、同じ種類のパネルを全パート分並べて比較できます。スペクトログラムの拡大範囲は連動します。
- スペクトログラムの明るさの基準は全パート共通です。パートの音量の差が、そのまま明るさの差になります。
- パート別の表示には、分離済みの解析結果が必要です。
  - 旧形式（0.1）の結果: ボーカルと全パートのみ
  - 単体版でファイルや YouTube を直接開いたとき: 全パートのみ

## 操作

### キーボード

ショートカットは、表示中のタブにだけ効きます。macOS では、この節の Ctrl をすべて ⌘ に読み替えてください（マウス操作の Ctrl も同じです）。

| キー | 歌声アナライザー | ビジュアライザー |
|---|---|---|
| Space | 再生 / 一時停止 | 再生 / 一時停止 |
| ← / → | 前 / 次のノート | 5 秒戻る / 進む |
| F11 | — | 全画面（Esc で戻る） |
| Ctrl+1 / Ctrl+2 | タブの切り替え（統合版） | タブの切り替え（統合版） |
| Ctrl+O | 解析結果を開く | 解析結果を開く（統合版） |

### マウス: ビジュアライザーの「スペクトログラム + ピッチ軌跡」

| 操作 | 動作 |
|---|---|
| ホイール | 拡大縮小（時間は再生位置を中心、周波数はマウスの位置を中心） |
| Shift + ホイール / 左の音名の上でホイール | 周波数だけ拡大縮小 |
| Ctrl + ホイール / 下の時刻の上でホイール | 時間だけ拡大縮小 |
| 左ドラッグ | 移動（上下 = 周波数、左右 = 再生位置の前後どちらを広く見るか） |
| 右ドラッグ | 拡大縮小（右・上へ動かすと拡大） |
| ダブルクリック | 元の範囲に戻す（過去 6 秒〜未来 2 秒、50Hz〜8kHz） |

### マウス: 歌声アナライザーのグラフ（音程バー、パート、スペクトログラムなど）

| 操作 | 動作 |
|---|---|
| ホイール | 拡大縮小（軸の上ではその軸だけ） |
| ドラッグ / 右ドラッグ | 移動 / 拡大縮小 |
| 左下の「A」ボタン | 全体表示に戻す |
| クリック | その位置へ移動（音程バーをクリックするとノートの詳細を表示） |

## コマンドラインオプション

```bash
.venv/bin/python -m musica.app --url "https://www.youtube.com/watch?v=XXXX" --start 60 --end 90
.venv/bin/python -m musica.app --result cache/results/<フォルダ> --tab visualizer
.venv/bin/python -m musica.app --visualizer --file song.mp3
```

| オプション | 内容 |
|---|---|
| `--analyzer` / `--visualizer` | 歌声アナライザー単体 / ビジュアライザー単体で起動 |
| `--tab analyzer\|visualizer` | 統合版で最初に開くタブ |
| `--url URL` | 起動時に YouTube から取得 |
| `--start 秒` / `--end 秒` | アナライザーで `--url` と使うと解析区間。ビジュアライザー単体では再生開始位置（`--end` は使わない）。`--result` と使うと、解析区間の先頭からの秒数 |
| `--result フォルダ` | 解析結果を開く |
| `--file ファイル` | 音声 / 動画ファイルを開く（ビジュアライザー単体） |
| `--source mix\|vocal` | 解析結果のどちらを鳴らすか（ビジュアライザー単体） |
| `--selftest [--network] [--demucs]` | 動作チェックをして終了 |

## 保存されるデータ

保存先は次のとおりです。環境変数 `MUSICA_CACHE` で変えられます。

- ソースから動かしているとき: `code/cache/`
- 実行ファイル版や、書き込めない場所に置いたとき: ユーザーデータの `MUSICA_EX/cache/`
  - Windows: `%LOCALAPPDATA%\MUSICA_EX`
  - macOS: `~/Library/Application Support/MUSICA_EX`
  - Linux: `~/.local/share/MUSICA_EX`

```
cache/
  downloads/<動画ID>.wav / .json        取得した音源 (44.1kHz ステレオ, 次回から再利用)
  results/<動画ID>_<開始>-<終了>_<分離方式>/
    analysis.json                       ノートと特徴量・コード・設定
    curves.npz                          ピッチ・音量・倍音などの時系列
    mix.wav                             区間の元の音
    stems/<パート>.wav                  分離したパート
    separation.json                     分離の設定 (同じ設定なら再利用)
  musica.log                            ログ (コンソールの無い起動時)
  selftest.txt                          動作チェックの結果
```

結果フォルダ名の開始・終了は 1/100 秒単位です（例: `XXXX_2000-6000_demucs_htdemucs` = 20.00〜60.00 秒）。

## 実行ファイル版（Python の無い PC に配る）

```bash
build_exe_windows.bat [--variant lite|cpu|cu126|cu130] [--zip]    # Windows
./build_exe.sh        [--variant lite|cpu|cu126|cu130] [--zip]    # Linux / macOS
```

| 種類 | 内容 |
|---|---|
| `lite`（既定） | Demucs なし（簡易分離のみ、小さい） |
| `cpu` | Demucs + CPU 版 PyTorch |
| `cu126` / `cu130` | Demucs + CUDA 版 PyTorch（非常に大きい） |

- 出力先は `code/dist/<種類>/MUSICA/` です。
  - `MUSICA`（統合版）、`MUSICA Analyzer`、`MUSICA Visualizer` の 3 つの実行ファイル（Windows は `.exe`）と README.txt ができます。3 つは同じ実行ファイルで、ファイル名で起動モードが変わります。
- ビルド後に動作チェックを自動で行います。`--zip` を付けると配布用の zip も作ります。
- その他のオプション: `--fresh` でビルド用の環境を作り直し、`--no-test` でビルド後の動作チェックを省略します。
- クロスコンパイルはできないので、Windows 用は Windows 上で作ります。
- GitHub Actions 用のワークフロー（[`code/.github/workflows/build-windows.yml`](code/.github/workflows/build-windows.yml)）もあります。ワークフローはリポジトリ直下の `.github/` にある必要があるため、`code` フォルダをリポジトリのルートにした場合に動きます。

## 動作チェックと困ったとき

- 動作チェックは次のどれかで実行できます。結果は `cache/selftest.txt` にも保存されます。
  - メニュー「ヘルプ → 動作チェック」
  - コマンド `--selftest`（YouTube への接続も見るなら `--network`、Demucs で実際に分離するなら `--demucs`）
  - Windows のスタートメニューの「MUSICA 動作チェック」
- **YouTube から取得できない**: yt-dlp を更新すると直ることが多いです。
  ```bash
  tools/uv/uv pip install --python .venv/bin/python -U "yt-dlp[default]"
  ```
  Windows では `tools\uv\uv.exe pip install --python .venv\Scripts\python.exe -U "yt-dlp[default]"` です。セットアップを実行し直すと、固定した版に戻ります。
- **GPU が使われない**: 「デバイス」の設定を確認してください。ドライバを更新したら、セットアップをもう一度実行すると合った PyTorch が入ります。GPU で失敗したときは自動で CPU でやり直します。
- **描画が遅い / おかしい**（リモートデスクトップ、仮想マシンなど）: `MUSICA_GL=0` を設定すると、GPU を使わずに描画します。ソフトウェア実装の OpenGL は自動で避けます。
- **音が出ない**: 音声の出力デバイスが無いときは、音を出さずに再生位置だけ進みます。
- **CPU での分離が遅い**: `MUSICA_CPU_THREADS` でスレッド数を指定できます（既定は論理 CPU の半分、最大 16）。

## 環境変数

| 変数 | 内容 |
|---|---|
| `MUSICA_CACHE` | データの保存先 |
| `MUSICA_GL=0` | GPU（OpenGL）を使わずに描画する |
| `MUSICA_CPU_THREADS` | CPU で Demucs を動かすときのスレッド数 |
| `MUSICA_SCREENSHOT` / `MUSICA_SCREENSHOT_DELAY` | 自動テスト用。起動して指定ミリ秒後（既定 6000）に画面を PNG に保存して終了する |

## フォルダ構成

```
code/
  musica_ex.py, vocal_analyzer.py, audio_visualizer.py   起動用 (統合版 / アナライザー単体 / ビジュアライザー単体)
  setup_windows.bat, setup.sh, uninstall_windows.bat    セットアップ / ショートカットの削除
  build_exe_windows.bat, build_exe.sh                   実行ファイル版を作る
  musica/
    app.py              起動処理 (起動モード・コマンドライン)
    analysis.py         解析の流れ (分離 → ピッチ → ノート → 特徴量 → コード → 保存)
    separation.py       音源分離 (Demucs / 簡易分離)
    pitch.py, notes.py, features.py, chords.py   ピッチ / 音程バー / 調教の特徴量 / コード・キー・テンポ
    youtube.py, audio_io.py                       YouTube の検索・取得 / 音声の読み書き (ffmpeg)
    gui/                main_window.py (統合版・アナライザー単体のウィンドウ), analyzer_window.py ほか
    visual/             ビジュアライザー (dsp.py = 信号処理, widgets.py = 描画パネル, window.py = 画面)
    selftest.py         動作チェック
  tools/                install.py (セットアップ本体), build_exe.py, musica.spec (PyInstaller)
  assets/               アイコン
```

YouTube などから取得した音源は、各サービスの利用規約と著作権の範囲で利用してください。
