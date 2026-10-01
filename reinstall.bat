@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo  Удаляю окружение .venv, при следующем запуске start.bat оно установится заново.
echo  Скачанные модели в hf_cache и сохранённые пациенты не пострадают.
pause
rmdir /s /q .venv
echo  Готово. Запустите start.bat
pause
