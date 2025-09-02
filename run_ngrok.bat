@echo off
setlocal

REM ngrok が PATH にない場合は、ngrok.exe を同フォルダに置くか
REM set PATH=%PATH%;C:\ngrok のように通してください。

REM 初回のみ：自分のトークンに置換して有効化（2回目以降はスキップされます）
ngrok config get authtoken >NUL 2>&1
if errorlevel 1 (
  ngrok config add-authtoken YOUR_NGROK_AUTH_TOKEN_HERE
)

REM 8000番にトンネル
ngrok http 8000
