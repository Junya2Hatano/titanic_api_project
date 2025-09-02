@echo off
setlocal

REM 仮想環境があれば有効化
if exist ".venv\Scripts\activate" (
  call .venv\Scripts\activate
)

REM 依存がなければ導入（保険）
python -m pip install --upgrade pip >NUL 2>&1
python -c "import uvicorn" 2>NUL || pip install -r requirements.txt

REM FastAPIサーバ起動（必要なら --host 0.0.0.0）
uvicorn main:app --host 127.0.0.1 --port 8000

pause
