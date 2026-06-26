#!/bin/bash
# PRISM v2 setup — textual UI + llama.cpp for 70B models
set -e

echo "=== PRISM v2 Setup ==="
echo "Platform: $(uname -m) | Python: $(python3 --version)"
echo ""

echo "Installing textual (terminal UI)..."
pip install "textual>=0.67.0" --quiet

echo "Installing llama-cpp-python with Metal GPU (70B support)..."
echo "(Compiles from source — 2-5 min)"
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --upgrade --quiet

echo "Installing huggingface_hub CLI..."
pip install "huggingface_hub[cli]>=0.23" --quiet

echo ""
echo "=== Verification ==="
python3 -c "
import textual; print(f'textual:          {textual.__version__}')
try:
    import llama_cpp; print(f'llama-cpp-python: {llama_cpp.__version__}')
except Exception as e:
    print(f'llama-cpp-python: WARN — {e}')
import mlx; print(f'mlx:              {mlx.__version__}')
import mlx_lm; print(f'mlx-lm:           {mlx_lm.__version__}')
"

echo ""
echo "=== Done ==="
echo ""
echo "Run UI:"
echo "  cd $(pwd)"
echo "  python prism_ui.py"
echo "  python prism_ui.py --model mlx-community/gemma-3-4b-it-4bit"
echo ""
echo "70B via GGUF (needs huggingface-cli download first):"
echo "  python prism_ui.py --gguf ./models/llama-70b/Meta-Llama-3.1-70B-Instruct-Q2_K.gguf"
echo ""
echo "Speculative decoding (2-4x SIMPLE speedup):"
echo "  python prism_ui.py --model mlx-community/gemma-3-4b-it-4bit --draft mlx-community/gemma-3-1b-it-4bit"
