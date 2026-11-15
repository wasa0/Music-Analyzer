@echo off
rem ----------------------------------------------------------------------
rem  Makes sure tools\uv\uv.exe exists (downloads uv and verifies SHA-256).
rem  Usage from another batch file:   call "%~dp0tools\get_uv.bat"
rem  Afterwards UV_EXE points to uv.exe. Returns errorlevel 1 on failure.
rem  No setlocal here on purpose: UV_EXE must stay visible to the caller.
rem  ASCII only (cmd.exe and Japanese text do not mix).
rem ----------------------------------------------------------------------
set "UV_VERSION=0.12.21"
set "UV_DIR=%~dp0uv"
set "UV_EXE=%~dp0uv\uv.exe"

set "UV_ARCH=%PROCESSOR_ARCHITECTURE%"
if defined PROCESSOR_ARCHITEW6432 set "UV_ARCH=%PROCESSOR_ARCHITEW6432%"
if /i "%UV_ARCH%"=="AMD64" goto :x64
if /i "%UV_ARCH%"=="ARM64" goto :arm64
echo This PC (%UV_ARCH%) is not supported. 64-bit Windows 10/11 is required.
exit /b 1

:x64
set "UV_TARGET=x86_64-pc-windows-msvc"
set "UV_SHA256=5d223efa0bf00208c3853246af09420419dfbd352536aa6bb8163d6170e23890"
goto :check

:arm64
set "UV_TARGET=aarch64-pc-windows-msvc"
set "UV_SHA256=93ed53b94e9cec000cacdfd18ca67bc4cb2b6a5f5ec041edd7f2a3dae365ce79"
goto :check

:check
if exist "%UV_EXE%" exit /b 0
echo [setup] Downloading uv %UV_VERSION% ...
set "UV_URL=https://github.com/astral-sh/uv/releases/download/%UV_VERSION%/uv-%UV_TARGET%.zip"
powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue';" ^
  "[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12;" ^
  "$zip=Join-Path $env:TEMP ('uv-musica-'+$PID+'.zip');" ^
  "Invoke-WebRequest -UseBasicParsing -Uri $env:UV_URL -OutFile $zip;" ^
  "$h=(Get-FileHash -Algorithm SHA256 -LiteralPath $zip).Hash.ToLower();" ^
  "if($h -ne $env:UV_SHA256){Remove-Item $zip; throw ('checksum mismatch: '+$h)};" ^
  "New-Item -ItemType Directory -Force -Path $env:UV_DIR | Out-Null;" ^
  "Expand-Archive -Force -LiteralPath $zip -DestinationPath $env:UV_DIR; Remove-Item $zip"
if errorlevel 1 (
  echo [setup] Could not download uv. Check your internet connection and try again.
  exit /b 1
)
if not exist "%UV_EXE%" (
  echo [setup] uv.exe was not found after download.
  exit /b 1
)
exit /b 0
