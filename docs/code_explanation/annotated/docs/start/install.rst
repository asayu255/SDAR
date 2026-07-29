.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Installation
============

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Requirements
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
------------

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- **Python**: Version >= 3.9
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- **CUDA**: Version >= 12.1

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
verl supports various backends. Currently, the following configurations are available:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- **FSDP** and **Megatron-LM** (optional) for training.
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- **SGLang**, **vLLM** and **TGI** for rollout generation.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Choices of Backend Engines
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
----------------------------

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
1. Training:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
We recommend using **FSDP** backend to investigate, research and prototype different models, datasets and RL algorithms. The guide for using FSDP backend can be found in :doc:`FSDP Workers<../workers/fsdp_workers>`.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
For users who pursue better scalability, we recommend using **Megatron-LM** backend. Currently, we support `Megatron-LM v0.11 <https://github.com/NVIDIA/Megatron-LM/tree/v0.11.0>`_. The guide for using Megatron-LM backend can be found in :doc:`Megatron-LM Workers<../workers/megatron_workers>`.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. note:: 

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    verl directly supports megatron's `GPTModel` API on the main branch with mcore v0.11. For mcore v0.4 try `0.3.x branch <https://github.com/volcengine/verl/tree/v0.3.x>`_ instead.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
2. Inference:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
For inference, vllm 0.6.3 and 0.8.2 have been tested for stability. Avoid using vllm 0.7x due to reported issues with its functionality.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
For SGLang, refer to the :doc:`SGLang Backend<../workers/sglang_worker>` for detailed installation and usage instructions. **SGLang offers better throughput and is under extensive development.** We encourage users to report any issues or provide feedback via the `SGLang Issue Tracker <https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/issues/106>`_.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
For huggingface TGI integration, it is usually used for debugging and single GPU exploration.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Install from docker image
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
-------------------------

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
We provide pre-built Docker images for quick setup.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
For vLLM with Megatron or FSDP, please use the stable version of image ``whatcanyousee/verl:ngc-cu124-vllm0.8.5-sglang0.4.6-mcore0.12.0-te2.3``.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
For latest vLLM with FSDP, please refer to ``hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0``.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
For SGLang with FSDP, please use ``ocss884/verl-sglang:ngc-th2.6.0-cu126-sglang0.4.6.post5`` which is provided by SGLang RL Group.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
See files under ``docker/`` for NGC-based image or if you want to build your own.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
1. Launch the desired Docker image and attach into it:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code:: bash

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" --cap-add=SYS_ADMIN -v .:/workspace/verl --name verl <image:tag>
    docker start verl
    docker exec -it verl bash


.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
2.	Inside the container, install latest verl:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code:: bash

    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # install the nightly version (recommended)
    git clone https://github.com/volcengine/verl && cd verl
    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # pick your choice of inference engine: vllm or sglang
    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # pip3 install -e .[vllm]
    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # pip3 install -e .[sglang]
    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # or install from pypi instead of git via:
    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # pip3 install verl[vllm]
    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # pip3 install verl[sglang]

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. note::

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    The Docker image ``whatcanyousee/verl:ngc-cu124-vllm0.8.5-sglang0.4.6-mcore0.12.0-te2.3`` is built with the following configurations:

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    - **PyTorch**: 2.6.0+cu124
    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    - **CUDA**: 12.4
    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    - **cuDNN**: 9.8.0
    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    - **nvidia-cudnn-cu12**: 9.8.0.87, **important for the usage of Megatron FusedAttention with MLA Support**
    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    - **Flash Attenttion**: 2.7.4.post1
    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    - **Flash Infer**: 0.2.2.post1
    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    - **vLLM**: 0.8.5
    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    - **SGLang**: 0.4.6.post5
    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    - **Megatron-LM**: core_v0.12.0
    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    - **TransformerEngine**: 2.3
    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    - **Ray**: 2.44.1

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. note::

   .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
   For aws instances with EFA net interface (Sagemaker AI Pod),
   you need to install EFA driver as shown in ``docker/Dockerfile.awsefa``

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Install from custom environment
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
---------------------------------------------

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
We recommend to use docker images for convinience. However, if your environment is not compatible with the docker image, you can also install verl in a python environment.


.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Pre-requisites
::::::::::::::

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
For training and inference engines to utilize better and faster hardware support, CUDA/cuDNN and other dependencies are required,
and some of the dependencies are easy to be overrided when installing other packages,
so we put them in the :ref:`Post-installation` step.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
We need to install the following pre-requisites:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- **CUDA**: Version >= 12.4
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- **cuDNN**: Version >= 9.8.0
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- **Apex**

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
CUDA above 12.4 is recommended to use as the docker image,
please refer to `NVIDIA's official website <https://developer.nvidia.com/cuda-toolkit-archive>`_ for other version of CUDA.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code:: bash

    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # change directory to anywher you like, in verl source code directory is not recommanded
    wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda-repo-ubuntu2204-12-4-local_12.4.1-550.54.15-1_amd64.deb
    dpkg -i cuda-repo-ubuntu2204-12-4-local_12.4.1-550.54.15-1_amd64.deb
    cp /var/cuda-repo-ubuntu2204-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/
    apt-get update
    apt-get -y install cuda-toolkit-12-4
    update-alternatives --set cuda /usr/local/cuda-12.4


.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
cuDNN can be installed via the following command,
please refer to `NVIDIA's official website <https://developer.nvidia.com/rdp/cudnn-archive>`_ for other version of cuDNN.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code:: bash

    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # change directory to anywher you like, in verl source code directory is not recommanded
    wget https://developer.download.nvidia.com/compute/cudnn/9.8.0/local_installers/cudnn-local-repo-ubuntu2204-9.8.0_1.0-1_amd64.deb
    dpkg -i cudnn-local-repo-ubuntu2204-9.8.0_1.0-1_amd64.deb
    cp /var/cudnn-local-repo-ubuntu2204-9.8.0/cudnn-*-keyring.gpg /usr/share/keyrings/
    apt-get update
    apt-get -y install cudnn-cuda-12

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
NVIDIA Apex is required for Megatron-LM and FSDP training.
You can install it via the following command, but notice that this steps can take a very long time.
It is recommanded to set the ``MAX_JOBS`` environment variable to accelerate the installation process,
but do not set it too large, otherwise the memory will be overloaded and your machines may hang.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code:: bash

    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # change directory to anywher you like, in verl source code directory is not recommanded
    git clone https://github.com/NVIDIA/apex.git && \
    cd apex && \
    MAX_JOB=32 pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./


.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Install dependencies
::::::::::::::::::::

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. note::

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    We recommend to use a fresh new conda environment to install verl and its dependencies.

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    **Notice that the inference frameworks often strictly limit your pytorch version and will directly override your installed pytorch if not paying enough attention.**

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    As a countermeasure, it is recommended to install inference frameworks first with the pytorch they needed. For vLLM, if you hope to use your existing pytorch,
    please follow their official instructions
    `Use an existing PyTorch installation <https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html#build-wheel-from-source>`_ .


.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
1. First of all, to manage environment, we recommend using conda:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code:: bash

   .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
   conda create -n verl python==3.10
   conda activate verl


.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
2. Then, execute the ``install.sh`` script that we provided in verl:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code:: bash

    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # Make sure you have activated verl conda env
    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # If you need to run with megatron
    bash scripts/install_vllm_sglang_mcore.sh
    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # Or if you simply need to run with FSDP
    USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh


.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
If you encounter errors in this step, please check the script and manually follow the steps in the script.


.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Install verl
::::::::::::

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
For installing the latest version of verl, the best way is to clone and
install it from source. Then you can modify our code to customize your
own post-training jobs.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code:: bash

   .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
   git clone https://github.com/volcengine/verl.git
   cd verl
   pip install --no-deps -e .


.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Post-installation
:::::::::::::::::

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Please make sure that the installed packages are not overridden during the installation of other packages.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
The packages worth checking are:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- **torch** and torch series
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- **vLLM**
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- **SGLang**
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- **pyarrow**
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- **tensordict**
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- **nvidia-cudnn-cu12**: For Magetron backend

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
If you encounter issues about package versions during running verl, please update the outdated ones.


.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Install with AMD GPUs - ROCM kernel support
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
------------------------------------------------------------------

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
When you run on AMD GPUs (MI300) with ROCM platform, you cannot use the previous quickstart to run verl. You should follow the following steps to build a docker and run it. 
If you encounter any issues in using AMD GPUs running verl, feel free to contact me - `Yusheng Su <https://yushengsu-thu.github.io/>`_.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Find the docker for AMD ROCm: `docker/Dockerfile.rocm <https://github.com/volcengine/verl/blob/main/docker/Dockerfile.rocm>`_
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code-block:: bash

    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    #  Build the docker in the repo dir:
    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # docker build -f docker/Dockerfile.rocm -t verl-rocm:03.04.2015 .
    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # docker images # you can find your built docker
    FROM rocm/vllm:rocm6.2_mi300_ubuntu20.04_py3.9_vllm_0.6.4

    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # Set working directory
    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # WORKDIR $PWD/app

    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # Set environment variables
    ENV PYTORCH_ROCM_ARCH="gfx90a;gfx942"

    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # Install vllm
    RUN pip uninstall -y vllm && \
        rm -rf vllm && \
        git clone -b v0.6.3 https://github.com/vllm-project/vllm.git && \
        cd vllm && \
        MAX_JOBS=$(nproc) python3 setup.py install && \
        cd .. && \
        rm -rf vllm

    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # Copy the entire project directory
    COPY . .

    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # Install dependencies
    RUN pip install "tensordict<0.6" --no-deps && \
        pip install accelerate \
        codetiming \
        datasets \
        dill \
        hydra-core \
        liger-kernel \
        numpy \
        pandas \
        datasets \
        peft \
        "pyarrow>=15.0.0" \
        pylatexenc \
        "ray[data,train,tune,serve]" \
        torchdata \
        transformers \
        wandb \
        orjson \
        pybind11 && \
        pip install -e . --no-deps

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Build the image
::::::::::::::::::::::::

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code-block:: bash

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    docker build -t verl-rocm .

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Launch the container
::::::::::::::::::::::::::::

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code-block:: bash

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    docker run --rm -it \
      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      --device /dev/dri \
      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      --device /dev/kfd \
      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      -p 8265:8265 \
      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      --group-add video \
      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      --cap-add SYS_PTRACE \
      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      --security-opt seccomp=unconfined \
      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      --privileged \
      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      -v $HOME/.ssh:/root/.ssh \
      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      -v $HOME:$HOME \
      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      --shm-size 128G \
      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      -w $PWD \
      verl-rocm \
      /bin/bash

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
(Optional): If you do not want to root mode and require assign yuorself as the user
Please add ``-e HOST_UID=$(id -u)`` and ``-e HOST_GID=$(id -g)`` into the above docker launch script. 

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
(Currently Support): Training Engine: FSDP; Inference Engine: vLLM and SGLang - We will support Megatron in the future.
