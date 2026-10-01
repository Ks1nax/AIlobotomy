@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title AIlobotomy

rem Модели скачиваются в папку hf_cache рядом с программой
set "HF_HOME=%~dp0hf_cache"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"

rem Версия CUDA для PyTorch. Для старых видеокарт GTX 900/1000 замените cu128 на cu126
set "TORCH_CUDA=cu128"

if exist ".venv\Scripts\python.exe" goto run

echo.
echo  Первый запуск: готовлю окружение, это займёт несколько минут
echo.
where py >nul 2>nul
if not errorlevel 1 (
  py -3 -m venv .venv
) else (
  python -m venv .venv
)
if not exist ".venv\Scripts\python.exe" (
  echo.
  echo  Не найден Python. Установите Python 3.10 или новее с python.org
  echo  и при установке отметьте галочку "Add python.exe to PATH".
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m pip install --upgrade pip

set "GPU="
where nvidia-smi >nul 2>nul
if not errorlevel 1 set "GPU=1"
if defined GPU (
  echo  Найдена видеокарта NVIDIA, ставлю PyTorch с поддержкой CUDA
  ".venv\Scripts\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/%TORCH_CUDA%
) else (
  echo  Видеокарта NVIDIA не найдена, ставлю PyTorch для процессора
  ".venv\Scripts\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo  Установка не удалась. Проверьте интернет и запустите start.bat ещё раз.
  rmdir /s /q .venv
  pause
  exit /b 1
)

:run
".venv\Scripts\python.exe" server.py
pause
