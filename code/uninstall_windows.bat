@echo off
rem Removes the desktop / Start menu shortcuts created by setup_windows.bat.
rem To remove the app itself, delete this folder afterwards.
setlocal EnableExtensions
cd /d "%~dp0"
set "UV_EXE=%~dp0tools\uv\uv.exe"
set "UV_PYTHON_INSTALL_DIR=%~dp0tools\python"
set "PYTHONUTF8=1"
if not exist "%UV_EXE%" (
  echo setup_windows.bat has not been run in this folder.
  if not defined MUSICA_NO_PAUSE pause
  exit /b 1
)
"%UV_EXE%" run --no-project --managed-python --python 3.12 "%~dp0tools\install.py" --uninstall
if not defined MUSICA_NO_PAUSE pause
