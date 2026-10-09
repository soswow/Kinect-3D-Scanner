@echo off
pushd "%~dp0"
"%~dp0.venv\Scripts\python.exe" -m scanner_server %*
set "server_exit=%errorlevel%"
if not "%server_exit%"=="0" pause
popd
exit /b %server_exit%
