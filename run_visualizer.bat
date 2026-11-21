@echo off
title URD-302 Interactive SNN Model Visualizer & Live Fine-Tuning Studio
echo ===============================================================================
echo URD-302 INTERACTIVE SNN MODEL VISUALIZER ^& LIVE FINE-TUNING STUDIO
echo Student : Ko Sett Paing (KOSETTPAI)
echo Advisor : Dr. Etienne Mueller
echo ===============================================================================
cd /d "%~dp0"
start "" "http://127.0.0.1:5050"
python app.py
pause
