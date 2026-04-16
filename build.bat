@echo off
REM Docker build and export script for Windows
REM Usage: build.bat team_name

if "%~1"=="" (
    echo Usage: build.bat team_name
    echo Example: build.bat team_alpha
    exit /b 1
)

set TEAM_NAME=%~1
set IMAGE_NAME=bdc2026
set OUTPUT_FILE=%TEAM_NAME%.tar

echo ========================================
echo Building Docker image
echo ========================================
echo Image name: %IMAGE_NAME%
echo Output file: %OUTPUT_FILE%
echo.

echo Building image...
docker build -t %IMAGE_NAME% .
if errorlevel 1 (
    echo Build failed!
    exit /b 1
)

echo.
echo Exporting image...
docker save -o %OUTPUT_FILE% %IMAGE_NAME%
if errorlevel 1 (
    echo Export failed!
    exit /b 1
)

echo.
echo ========================================
echo Build completed successfully!
echo ========================================
dir %OUTPUT_FILE%
echo.
echo Submit file: %OUTPUT_FILE%
