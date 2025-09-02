@echo off
setlocal

echo === FastAPI サーバーと ngrok を同時に起動します ===

REM 仮想環境があれば有効化
if exist ".venv\Scripts\activate" (
  call .venv\Scripts\activate
)

REM 必要なライブラリが入っているか簡単チェック
python -c "import uvicorn" 2>NUL || pip install -r requirements.txt

REM FastAPI サーバーをバックグラウンドで起動
start cmd /k uvicorn main:app --host 127.0.0.1 --port 8000

REM ngrok トンネルを起動（8000番ポート）
REM 初回は authtoken を追加する必要あり
start cmd /k ngrok http 8000

echo.
echo サーバーと ngrok を起動しました。
echo ngrok のウィンドウに表示される「Forwarding」の https://～ を
echo Bubble の API Connector に設定してください。
echo.
pause
