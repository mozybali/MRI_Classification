@echo off
setlocal

set "ROOT=%~dp0"
set "VENV_PYTHON=%ROOT%.venv\Scripts\python.exe"
set "REQ_FILE=%ROOT%requirements.txt"

if not exist "%VENV_PYTHON%" (
    echo [venv] .venv bulunamadi, olusturuluyor...
    where py >nul 2>nul
    if %ERRORLEVEL%==0 (
        py -3 -m venv "%ROOT%.venv"
    ) else (
        python -m venv "%ROOT%.venv"
    )
    if errorlevel 1 (
        echo [hata] Sanal ortam olusturulamadi.
        exit /b 1
    )
)

echo [venv] Yeni PowerShell oturumu aciliyor...
echo [venv] Cikmak icin: deactivate

powershell -NoExit -ExecutionPolicy Bypass -Command ^
    "Set-Location '%ROOT%'; " ^
    ". '.\.venv\Scripts\Activate.ps1'; " ^
    "if (Test-Path '%REQ_FILE%') { Write-Host '[venv] Gerekirse paket kur: pip install -r requirements.txt' }"
