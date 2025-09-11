@echo off
echo Installing PyTorch with CUDA support...
E:/dev/diffusers-try/.venv/Scripts/pip.exe uninstall torch torchvision torchaudio -y
E:/dev/diffusers-try/.venv/Scripts/pip.exe install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo.
echo Testing CUDA...
E:/dev/diffusers-try/.venv/Scripts/python.exe test_cuda.py
pause
