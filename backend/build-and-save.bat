@echo off
REM build-and-save.bat - Build Octavia Docker image and save for sharing
REM Run this from the backend folder: build-and-save

echo ============================================
echo Octavia Docker Build Script
echo ============================================
echo.

set DOCKERFILE=Dockerfile.with-ollama
set IMAGE_NAME=octavia-with-ollama
set OUTPUT_FILE=octavia-with-ollama.tar

echo Building image: %IMAGE_NAME%
echo Using: %DOCKERFILE%
echo.
echo Estimated time: 60-90 minutes on 10 Mbps
echo.
echo Press Ctrl+C to cancel, or any key to continue...
pause > nul

echo.
echo Starting build...
echo.

docker build -f %DOCKERFILE% -t %IMAGE_name% .

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Build failed!
    exit /b 1
)

echo.
echo Build successful!
echo.
echo Saving image to %OUTPUT_FILE%...
echo (This may take a few minutes depending on file size)

docker save %IMAGE_NAME% -o %OUTPUT_FILE%

echo.
echo ============================================
echo Build complete!
echo.
echo Image saved to: %OUTPUT_FILE%
echo Size: %~z1 bytes
echo.
echo To share: Copy %OUTPUT_FILE% to another machine
echo.
echo To load on another machine:
echo   docker load -i %OUTPUT_FILE%
echo   docker run -p 8000:8000 -p 11434:11434 %IMAGE_NAME%
echo ============================================
