@echo off
:: ============================================================
:: GNN Thesis -- Environment Setup (Windows, Python 3.13 uyumlu)
:: ============================================================
chcp 65001 >nul

echo.
echo ============================================================
echo   GNN Thesis -- Environment Kurulumu
echo ============================================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [HATA] Python bulunamadi.
    pause
    exit /b 1
)

echo Python versiyonu:
python --version

echo.
echo [1/6] Eski .venv temizleniyor (varsa)...
if exist .venv (
    rmdir /s /q .venv
    echo       Eski .venv silindi.
) else (
    echo       .venv yok, atlaniyor.
)

echo.
echo [2/6] Virtual environment olusturuluyor...
python -m venv .venv
echo       .venv olusturuldu.

echo.
echo [3/6] Aktif ediliyor ve pip guncelleniyor...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel

echo.
echo [4/6] Temel bilimsel paketler yukleniyor...
pip install numpy pandas scipy scikit-learn

echo.
echo [5/6] Derin ogrenme ve diger paketler yukleniyor...
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install xgboost
pip install statsmodels
pip install networkx
pip install matplotlib seaborn
pip install openpyxl requests beautifulsoup4
pip install ipykernel jupyter
pip install black isort pytest

echo.
echo [6/6] Kurulum kontrol ediliyor...
python -c "import numpy;  print('  numpy    :', numpy.__version__)"
python -c "import pandas; print('  pandas   :', pandas.__version__)"
python -c "import torch;  print('  torch    :', torch.__version__)"
python -c "import sklearn;print('  sklearn  :', sklearn.__version__)"
python -c "import xgboost;print('  xgboost  :', xgboost.__version__)"

echo.
echo ============================================================
echo   KURULUM TAMAMLANDI
echo ============================================================
echo.
echo VS Code adimi:
echo   1. GNN_Thesis.code-workspace dosyasina cift tikla
echo   2. Ctrl+Shift+P
echo   3. "Python: Select Interpreter" yaz, Enter
echo   4. Listeden  .venv\Scripts\python.exe  sec
echo   5. configs\config.py dosyasindaki 3 yolu guncelle
echo.
pause
