@echo off
echo Installing FFmpeg for video export...

REM Check if ffmpeg already exists
where ffmpeg >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo FFmpeg is already installed and in PATH
    ffmpeg -version
    pause
    exit /b 0
)

echo FFmpeg not found in PATH. Attempting to download...

REM Create bin directory if it doesn't exist
if not exist "bin" mkdir bin

REM Download FFmpeg (Windows 64-bit)
echo Downloading FFmpeg...
powershell -Command "(New-Object Net.WebClient).DownloadFile('https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip', 'ffmpeg.zip')"

if exist "ffmpeg.zip" (
    echo Extracting FFmpeg...
    powershell -Command "Expand-Archive -Path 'ffmpeg.zip' -DestinationPath '.' -Force"
    
    REM Move ffmpeg.exe to bin directory
    for /d %%i in (ffmpeg-master-latest-win64-gpl*) do (
        copy "%%i\bin\ffmpeg.exe" "bin\ffmpeg.exe"
        copy "%%i\bin\ffprobe.exe" "bin\ffprobe.exe"
    )
    
    REM Cleanup
    del ffmpeg.zip
    for /d %%i in (ffmpeg-master-latest-win64-gpl*) do rd /s /q "%%i"
    
    if exist "bin\ffmpeg.exe" (
        echo ✅ FFmpeg installed successfully to bin\ffmpeg.exe
        echo You can now use video export in the preview viewer!
        bin\ffmpeg.exe -version
    ) else (
        echo ❌ Failed to install FFmpeg
    )
) else (
    echo ❌ Failed to download FFmpeg
    echo Please install FFmpeg manually:
    echo 1. Download from: https://ffmpeg.org/download.html
    echo 2. Extract to bin\ folder
    echo 3. Or add to system PATH
)

pause
