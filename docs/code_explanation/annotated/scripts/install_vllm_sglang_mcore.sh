#!/bin/bash

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
USE_MEGATRON=${USE_MEGATRON:-1}
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
USE_SGLANG=${USE_SGLANG:-1}

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
export MAX_JOBS=32

# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
echo "1. install inference frameworks and pytorch they need"
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
if [ $USE_SGLANG -eq 1 ]; then
    # [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
    pip install "sglang[all]==0.4.6.post1" --no-cache-dir --find-links https://flashinfer.ai/whl/cu124/torch2.6/flashinfer-python && pip install torch-memory-saver --no-cache-dir
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
fi
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
pip install --no-cache-dir "vllm==0.8.5.post1" "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" "tensordict==0.6.2" torchdata

# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
echo "2. install basic packages"
# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
pip install "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas \
    ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
    pytest py-spy pyext pre-commit ruff

# [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"


# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
echo "3. install FlashAttention and FlashInfer"
# Install flash-attn-2.7.4.post1 (cxx11abi=False)
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl && \
    pip install --no-cache-dir flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Install flashinfer-0.2.2.post1+cu124 (cxx11abi=False)
# vllm-0.8.3 does not support flashinfer>=0.2.3
# see https://github.com/vllm-project/vllm/pull/15777
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
wget -nv https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.2.post1/flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl && \
    pip install --no-cache-dir flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl


# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
if [ $USE_MEGATRON -eq 1 ]; then
    # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
    echo "4. install TransformerEngine and Megatron"
    # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
    echo "Notice that TransformerEngine installation can take very long time, please be patient"
    # [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
    NVTE_FRAMEWORK=pytorch pip3 install --no-deps git+https://github.com/NVIDIA/TransformerEngine.git@v2.2
    # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
    pip3 install --no-deps git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.12.0rc3
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
fi


# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
echo "5. May need to fix opencv"
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
pip install opencv-python
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
pip install opencv-fixer && \
    python -c "from opencv_fixer import AutoFix; AutoFix()"


# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
if [ $USE_MEGATRON -eq 1 ]; then
    # [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
    echo "6. Install cudnn python package (avoid being overrided)"
    # [EXPLAIN] 実行設定または後続 command が参照する環境値・引数を定義する。
    pip install nvidia-cudnn-cu12==9.8.0.87
# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
fi

# [EXPLAIN] 実験起動、環境準備または検証 command の一段階を実行する。
echo "Successfully installed all packages"
