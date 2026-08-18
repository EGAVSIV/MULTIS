@echo off
cd /d "%~dp0backend"
echo ============================================
echo MULTIS BACKGROUND SCANNER
echo Reading COPIEDDATA and generating results...
echo ============================================
python -m pip install -r requirements.txt
python run_scanners.py
echo.
echo Finished. Press any key to close.
pause
