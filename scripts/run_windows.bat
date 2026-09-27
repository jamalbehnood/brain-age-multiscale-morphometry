@echo off
setlocal
cd /d "%~dp0.."
python src\brain_age_multiscale.py
if errorlevel 1 exit /b %errorlevel%
endlocal
