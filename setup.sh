#!/bin/bash
set -e
echo "=== PRISM Local Setup ==="
echo "Installing MLX + dependencies for Apple Silicon..."
pip3 install --quiet mlx mlx-lm huggingface_hub psutil
echo "Verifying MLX..."
python3 -c "import mlx.core as mx; print(f'MLX version: {mx.__version__}'); print(f'Device: {mx.default_device()}')"
python3 -c "import mlx_lm; print('mlx-lm: OK')"
python3 -c "import psutil; print(f'psutil: OK, RAM: {psutil.virtual_memory().total // 1024//1024} MB')"
echo "=== Setup complete ==="
