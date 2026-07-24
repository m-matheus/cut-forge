@echo off
REM CutForge launcher — runs the app as a local web app in your browser.
REM Double-click this file to start. No PyInstaller/.exe needed.
REM
REM It uses the project's .venv (created with `python -m venv .venv`).
REM Requires dependencies installed:  .venv\Scripts\pip install -e .

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [CutForge] .venv nao encontrado.
  echo Crie com:  python -m venv .venv  ^&^&  .venv\Scripts\pip install -e .
  pause
  exit /b 1
)

echo [CutForge] Iniciando... uma aba do navegador vai abrir em instantes.
".venv\Scripts\python.exe" -m cutforge.ui.desktop --browser
