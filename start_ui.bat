@echo off
echo Installing UI dependencies...
E:\dev\diffusers-try\.venv\Scripts\pip.exe install -r ui_requirements.txt

echo Starting Diffusers UI...
E:\dev\diffusers-try\.venv\Scripts\python.exe diffusers_ui.py

pause
