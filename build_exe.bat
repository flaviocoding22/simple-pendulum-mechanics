@echo off
REM Gera dist\Pendulo Simples.exe a partir de pendulo-simples\pendulo_v8.py
REM Pre-requisito (uma vez):  pip install pyinstaller
cd /d "%~dp0"

pyinstaller --onefile --windowed --noconfirm --clean ^
  --name "Pendulo Simples" ^
  --icon "pendulo-simples\pendulo.ico" ^
  --hidden-import matplotlib.backends.backend_qtagg ^
  "pendulo-simples\pendulo_v8.py"

echo.
echo Pronto: dist\Pendulo Simples.exe
pause
