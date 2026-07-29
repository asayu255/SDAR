.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
=======================
Search Tool Integration
=======================
Introduction
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
------------
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- We have added a search tool calling function to Multi-Turn RL, enabling the model to initiate retrieval requests during Actor rollout and directly use retrieval results for training. **We support using a local dense retriever as the retrieval tool, as well as integrating with your own local retrieval engine.**



.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Quick Reproduction
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
------------------

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Create a New Docker Container
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   docker run \
       -it \
       --shm-size 32g \
       --gpus all \
       -v {Huggingface-Cache-Path}:/root/.cache \
       --ipc=host \
       --network=host \
       --privileged \
       --name sglang_{your-name} \
       lmsysorg/sglang:dev \
       /bin/zsh

If you need to restart after exiting the container:

.. code:: bash

   docker start -i sglang_{your-name}

Update Python and Configure the Virtual Environment using uv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code:: bash

   .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
   apt update
   apt install -y python3.10 python3.10-venv

   .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
   # Create a virtual environment
   python3 -m venv ~/.python/verl-multiturn-rollout

   .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
   # Activate the virtual environment
   source ~/.python/verl-multiturn-rollout/bin/activate

   .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
   # Install uv
   python3 -m pip install uv

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Install verl Upstream
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   cd ~
   git clone https://github.com/volcengine/verl.git
   cd verl

   # Install verl
   python3 -m uv pip install .
   python3 -m uv pip install -r ./requirements_sglang.txt

   # Manually install flash-attn
   python3 -m uv pip install wheel
   python3 -m uv pip install packaging
   python3 -m uv pip install flash-attn --no-build-isolation --no-deps

Set Up a Local Retrieval Engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
If you are using your own local retrieval service, you can skip this
step. We chose the local dense retriever provided in the search-R1
example; detailed instructions are in the `searchR1
docs <https://raw.githubusercontent.com/PeterGriffinJin/Search-R1/refs/heads/main/docs/retriever.md>`__.
In brief:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
-  The GPU version offers higher accuracy and speed; each GPU uses about
   5–7 GB of memory.
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
-  The CPU version can be used for simple testing but has lower
   retrieval precision, which will degrade training performance. See the
   `retriever
   documentation <https://github.com/PeterGriffinJin/Search-R1/blob/main/docs/retriever.md>`__
   in search-R1 for details.
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
-  Recommend using Conda to install faiss-gpu=1.8.0; venv may cause errors.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
**Note**: To start both the training process and the local retrieval
service, we launch two separate Python environments. The training uses
uv in the verl-multiturn-rollout environment, while the retriever uses
conda to install ``faiss-gpu``.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code:: bash

   .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
   # Download the Miniconda installer script
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh

   .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
   # Install to $HOME/miniconda3 in batch mode
   bash ~/miniconda.sh -b -p $HOME/miniconda3

   .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
   # Activate conda (only in the current shell)
   eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

   .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
   # (Optional) Add conda to your default shell startup
   conda init

   .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
   # Reload shell config
   source ~/.bashrc

   .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
   # Create and activate the retriever environment with Python 3.10
   conda create -n retriever python=3.10 -y
   conda activate retriever

   .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
   # Install PyTorch (with GPU support) and related libraries
   conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y

   .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
   # Install other Python packages
   pip install transformers datasets pyserini huggingface_hub

   .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
   # Install the GPU version of faiss
   conda install faiss-gpu=1.8.0 -c pytorch -c nvidia -y

   .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
   # Install the API service framework
   pip install uvicorn fastapi

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Download the Indexing and Corpus
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The local retrieval files are large—prepare sufficient disk space.
Downloading is about 60–70 GB, and uncompressed takes about 132 GB:

.. code:: bash

   conda activate retriever

   save_path=/the/path/to/save
   python examples/sglang_multiturn/search_r1_like/local_dense_retriever/download.py --save_path $save_path
   cat $save_path/part_* > $save_path/e5_Flat.index
   gzip -d $save_path/wiki-18.jsonl.gz

Start the Local flat e5 Retrieval Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
1. The first startup will download models and load the index.
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
2. Apart from the download, startup takes about 1–2 minutes.
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
3. After startup, each GPU uses about 5–7 GB of memory, leaving the rest
   for multi-turn RL training.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code:: bash

   .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
   conda activate retriever

   .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
   index_file=$save_path/e5_Flat.index
   corpus_file=$save_path/wiki-18.jsonl
   retriever_name=e5
   retriever_path=intfloat/e5-base-v2

   .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
   python examples/sglang_multiturn/search_r1_like/local_dense_retriever/retrieval_server.py \
     .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
     --index_path $index_file \
     .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
     --corpus_path $corpus_file \
     .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
     --topk 3 \
     .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
     --retriever_name $retriever_name \
     .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
     --retriever_model $retriever_path \
     .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
     --faiss_gpu

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Set Up WANDB_API_KEY
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   export WANDB_API_KEY={YOUR_WANDB_API_KEY}

   # Define a timestamp function
   function now() {
       date '+%Y-%m-%d-%H-%M'
   }

**Preprocess the Dataset**
~~~~~~~~~~~~~~~~~~~~~~~~~~

   .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
   **Note:** The following data processing and training commands must be
   run in the verl-multiturn-rollout environment.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code:: bash

   .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
   python3 examples/data_preprocess/preprocess_search_r1_dataset.py

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Testing on 8 x H20
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
~~~~~~~~~~~~~~~~~~

.. code:: bash

   # Ensure the now() function is defined
   # Create a logs directory
   mkdir -p logs

   # Set GPUs and run with a suitable log path
   export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

   nohup bash examples/sglang_multiturn/search_r1_like/run_qwen2.5-3b_instruct_search_multiturn.sh \
     trainer.experiment_name=qwen2.5-3b-it_rm-searchR1-like-sgl-multiturn-$(now) \
     > logs/searchR1-like$(now).log 2>&1 &

Custom Search Configuration
---------------------------

To enable multi-turn reasoning, set the following fields in your config:

.. code:: yaml

   actor_rollout_ref:
     rollout:
       name: "sglang_async"
       multi_turn:
         enable: True

You must specify ``retrieval_service_url`` in ``examples/sglang_multiturn/config/tool_config/search_tool_config.yaml``, and properly configure concurrency. For more details on concurrency, refer to the Sandbox Fusion example:

.. code:: yaml

   tools:
     - class_name: verl.tools.search_tool.SearchTool
       config:
         retrieval_service_url: http://127.0.0.1:8000/retrieve
         num_workers: 120
         rate_limit: 120
         timeout: 30

The retriever input/output formats are as follows. If your service
parameters match, only modify ``retrieval_service_url``. You can also
customize in ``search_r1_like_utils.py``.

.. code:: python

   Input format:
   {
     "queries": ["What is Python?", "Tell me about neural networks."],
     "topk": 3,
     "return_scores": true
   }

   Output format (when return_scores=True, similarity scores are returned):
   {
       "result": [
           [   # Results for each query
               {
                   "document": doc, "score": score
               },
               # ... more documents
           ],
           # ... results for other queries
       ]
   }

Notes
-----

1. The total training time is about 27 hours; meanwhile, the validation
   dataset is very large (51 k), and each validation takes about 6000 s.
   (Therefore, ``val_before_train=False`` by default)
