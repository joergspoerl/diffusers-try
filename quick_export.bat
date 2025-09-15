@echo off
REM Standalone Video Exporter - Windows Batch Wrapper
REM ================================================

echo 🎬 Video Exporter - Quick Start
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python not found! Please install Python 3.6+
    pause
    exit /b 1
)

REM Check if video_exporter.py exists
if not exist "video_exporter.py" (
    echo ❌ Error: video_exporter.py not found in current directory
    pause
    exit /b 1
)

echo Available options:
echo.
echo 1. Quick Export (30 FPS, medium quality)
echo 2. High Quality (60 FPS, high quality, 2x upscale)
echo 3. YouTube Optimized (60 FPS, YouTube preset)
echo 4. With Interpolation (30→60 FPS, MCI mode)
echo 5. Custom Parameters
echo.

set /p choice="Choose option (1-5): "

if "%choice%"=="1" goto quick
if "%choice%"=="2" goto hq
if "%choice%"=="3" goto youtube
if "%choice%"=="4" goto interpolation
if "%choice%"=="5" goto custom
goto invalid

:quick
set /p input="Input folder: "
set /p output="Output file (e.g., video.mp4): "
echo 🚀 Creating quick video...
python video_exporter.py "%input%" "%output%"
goto end

:hq
set /p input="Input folder: "
set /p output="Output file (e.g., video.mp4): "
echo 🚀 Creating high quality video...
python video_exporter.py "%input%" "%output%" --fps 60 --quality high --upscale 2x
goto end

:youtube
set /p input="Input folder: "
set /p output="Output file (e.g., video.mp4): "
echo 🚀 Creating YouTube optimized video...
python video_exporter.py "%input%" "%output%" --fps 60 --quality youtube
goto end

:interpolation
set /p input="Input folder: "
set /p output="Output file (e.g., video.mp4): "
echo 🚀 Creating video with interpolation...
python video_exporter.py "%input%" "%output%" --fps 30 --interpolation-fps 60 --interpolation-mode mci
goto end

:custom
set /p input="Input folder: "
set /p output="Output file (e.g., video.mp4): "
set /p params="Additional parameters: "
echo 🚀 Creating video with custom parameters...
python video_exporter.py "%input%" "%output%" %params%
goto end

:invalid
echo ❌ Invalid choice!
goto end

:end
echo.
echo Press any key to exit...
pause >nul
