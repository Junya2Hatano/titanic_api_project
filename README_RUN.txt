【前提】
- Windows 11 / Python 3.11 以上推奨
- このフォルダを任意の場所に展開してください（パスに日本語・空白を含まない方が安全）

【初期セットアップ】（初回のみ）
1) Pythonをインストール（PATHに追加）
2) コマンドプロンプトで program フォルダへ移動
3) 仮想環境を作成して有効化（任意）
   python -m venv .venv
   .venv\Scripts\activate
4) 依存をインストール
   pip install --upgrade pip
   pip install -r requirements.txt

【起動】
A) ローカルのみで試す：
   1) run_server.bat をダブルクリック
   2) ブラウザで http://127.0.0.1:8000/docs を開く（API動作確認）
   3) Bubble をローカルAPIに向ける場合は、API ConnectorのベースURLを http://127.0.0.1:8000 に変更

B) 外部端末からも見せたい（ngrok使用）：
   1) run_server.bat を起動
   2) 別ウィンドウで run_ngrok.bat を起動
   3) 表示された Forwarding の https://xxxxx.ngrok-free.app を README_URL.txt の API ベースURLとして利用

【終了】
- それぞれのウィンドウで Ctrl + C で停止
