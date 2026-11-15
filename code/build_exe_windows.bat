@echo off
rem ======================================================================
rem  Builds the portable Windows version: dist\<variant>\MUSICA\MUSICA.exe
rem    build_exe_windows.bat                 (lite: no Demucs, small)
rem    build_exe_windows.bat --variant cpu   (Demucs + CPU PyTorch)
rem    build_exe_windows.bat --variant cu130 (Demucs + CUDA 13 PyTorch, very large)
rem    add --zip to also create a zip for sharing
rem  ASCII only on purpose (cmd.exe and Japanese text do not mix).
rem ======================================================================
setlocal EnableExtensions
cd /d "%~dp0"
call "%~dp0tools\get_uv.bat"
if errorlevel 1 goto :fail
set "UV_PYTHON_INSTALL_DIR=%~dp0tools\python"
set "MUSICA_UV=%UV_EXE%"
set "PYTHONUTF8=1"
"%UV_EXE%" run --no-project --managed-python --python 3.12 "%~dp0tools\build_exe.py" %*
if errorlevel 1 goto :fail
if not defined MUSICA_NO_PAUSE pause
exit /b 0

:fail
echo.
echo [build] Build did not finish. Please read the messages above.
if not defined MUSICA_NO_PAUSE pause
exit /b 1
