@echo off
:: ── Worm Counter Launcher (Windows) ────────────────────────────────────────
:: Double-click this file to start the app.
:: Keep this window open while using the app. Close it to stop the server.

cd /d "%~dp0"
set APP_DIR=%~dp0worm_app

echo ============================================
echo   Worm Counter
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found.
    echo.
    echo Please install Python 3 from: https://www.python.org/downloads/
    echo IMPORTANT: During install, check "Add Python to PATH"
    echo.
    pause
    exit /b 1
)

:: Install dependencies if missing
echo Checking dependencies...
python -c "import cv2, flask, flask_cors, numpy, PIL" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages - please wait...
    pip install opencv-python-headless flask flask-cors numpy Pillow
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to install packages. Check your internet connection.
        pause
        exit /b 1
    )
    echo Packages installed successfully.
)
echo Dependencies OK.
echo.

:: Kill any existing instance on port 8080
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":8080 "') do (
    taskkill /PID %%a /F >nul 2>&1
)

:: Add Windows Firewall rule to allow port 8080 (enables network access from other computers)
netsh advfirewall firewall show rule name="Worm Counter" >nul 2>&1
if errorlevel 1 (
    echo Adding firewall rule for network access...
    netsh advfirewall firewall add rule name="Worm Counter" dir=in action=allow protocol=TCP localport=8080 >nul 2>&1
)

:: Get local IP address for sharing
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /i "IPv4"') do (
    set LOCAL_IP=%%a
    goto :got_ip
)
:got_ip
set LOCAL_IP=%LOCAL_IP: =%

:: Open browser after a 2-second delay (runs in background so Flask can start first)
echo Opening browser in 2 seconds...
start /b cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:8080"

:: Start Flask in the FOREGROUND — keep this window open while app is running
echo Starting server...
echo.
echo   Your address:      http://127.0.0.1:8080
echo   Share with others: http://%LOCAL_IP%:8080
echo.
echo *** Keep this window open while using the app ***
echo *** Close this window (or press Ctrl+C) to stop ***
echo.
cd /d "%APP_DIR%"
python app.py

echo.
echo Server stopped.
pause
