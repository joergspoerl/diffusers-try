@echo off
echo Repariere NumPy/OpenCV Kompatibilitaet...
echo.

REM Deinstalliere problematische Versionen
echo Deinstalliere inkompatible Versionen...
pip uninstall numpy opencv-python -y

REM Installiere kompatible Versionen
echo.
echo Installiere NumPy 1.x...
pip install "numpy>=1.21.0,<2.0"

echo.
echo Installiere kompatible OpenCV Version...
pip install "opencv-python>=4.6.0,<4.10.0"

echo.
echo Pruefe Versionen...
pip show numpy
echo.
pip show opencv-python

echo.
echo Teste Imports...
python -c "import numpy; print(f'NumPy {numpy.__version__} OK')"
python -c "import cv2; print(f'OpenCV {cv2.__version__} OK')"

echo.
echo Kompatibilitaetsreparatur abgeschlossen!
pause
