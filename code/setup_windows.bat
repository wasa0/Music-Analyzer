@echo off
rem ======================================================================
rem  MUSICA_EX setup for Windows (double-click this file)
rem    - downloads uv (Python package manager, checksum verified)
rem    - uv downloads Python 3.12 into tools\python (no system Python needed)
rem    - tools\install.py creates .venv, installs libraries, chooses PyTorch
rem      for your GPU and creates desktop / Start menu shortcuts
rem  Options (from a terminal): setup_windows.bat --gpu cpu | --gpu none
rem  This file is ASCII only on purpose (cmd.exe and Japanese text do not mix).
rem ======================================================================
setlocal EnableExtensions
cd /d "%~dp0"
call "%~dp0tools\get_uv.bat"
if errorlevel 1 goto :fail

rem Keep Python inside this folder so deleting the folder removes everything.
set "UV_PYTHON_INSTALL_DIR=%~dp0tools\python"
set "MUSICA_UV=%UV_EXE%"
set "PYTHONUTF8=1"
"%UV_EXE%" run --no-project --managed-python --python 3.12 "%~dp0tools\install.py" %*
if errorlevel 1 goto :fail
echo.
echo [setup] Done. Start the app from the desktop shortcut "MUSICA".
if not defined MUSICA_NO_PAUSE pause
exit /b 0

:fail
echo.
echo [setup] Setup did not finish. Please read the messages above.
if not defined MUSICA_NO_PAUSE pause
exit /b 1
