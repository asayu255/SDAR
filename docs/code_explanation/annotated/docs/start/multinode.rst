.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Multinode Training
==================

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. _wuxibin89: https://github.com/wuxibin89

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Author: `Xibin Wu <https://github.com/wuxibin89>`_, `Yusheng Su <https://yushengsu-thu.github.io/>`_.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Manual
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
------

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Set up multinode ray cluster
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Start head node with ``ray start --head --dashboard-host=0.0.0.0``, there're 2 address you should care about:

- GCS address: ``ray start --address=<address>``, where worker node should connect to.
- Dashboard address: ``<address>:8265``, where you should submit job to the cluster.

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/head.png?raw=true

2. Start worker node with ``ray start --address=<address>`` you get above.

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/worker.png?raw=true

3. Now you should see the cluster have 2 nodes with ``ray status``.

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/status.png?raw=true

4. Additionally, you can access dashboard in the browser with the address you get above. 

*Firewall rules maybe need configure to access the dashboard, if there's any trouble, please contact your network administrator.*

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/overview.png?raw=true

Submit job to ray cluster
~~~~~~~~~~~~~~~~~~~~~~~~~
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
1. Submit ray job to cluster with the dashboard address you get above.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code-block:: bash

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    ray job submit --address="http://127.0.0.1:8265" \
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        --runtime-env=verl/trainer/runtime_env.yaml \
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        --no-wait \
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        -- \
        python3 -m verl.trainer.main_ppo \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=2 \
        ...

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/submit.png?raw=true

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
2. Then you can check the job status with the following commands:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- ray job list: list all jobs submitted to the cluster.
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- ray job logs <Submission ID>: query the logs of the job.
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- ray job status <Submission ID>: query the status of the job.
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
- ray job stop <Submission ID>: request the job to be stopped.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
3. You can also access driver/task/actor logs in ``/tmp/ray/session_latest/logs/``, driver log is ``job-driver-raysubmit_<Submission ID>.log``.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
4. We strongly recommend you to view job detail from dashboard in multinode training, because it provide more structure way to view the job information.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/job.png?raw=true
.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/job_detail.png?raw=true

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. note:: 

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    From Ray 2.20, ``ray job submit`` or ``client = JobSubmissionClient("http://127.0.0.1:8265")`` is deprecated in current environment, and Ray version less than 2.40 is not compatible with current version of verl. We recommend you upgrade to Ray latest version and directly execute the training scripts.


.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Slurm
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
-----
TBD

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
dstack
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
------
`dstackai/dstack <https://github.com/dstackai/dstack>`_ is an open-source container orchestrator that simplifies distributed training across cloud providers and on-premises environments
without the need to use K8S or Slurm.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Prerequisite
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
~~~~~~~~~~~~
Once dstack is `installed <https://dstack.ai/docs/installation>`_, initialize the directory as a repo with ``dstack init``. 

.. code-block:: bash

    mkdir myproject && cd myproject
    dstack init

**Create a fleet**

Before submitting distributed training jobs, create a `dstack` `fleet <https://dstack.ai/docs/concepts/fleets>`_.

Run a Ray cluster task
~~~~~~~~~~~~~~~~~~~~~~

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Once the fleet is created, define a Ray cluster task, e.g. in ``ray-cluster.dstack.yml``:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code-block:: yaml

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    type: task
    name: ray-verl-cluster

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    nodes: 2

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    env:
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        - WANDB_API_KEY
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        - PYTHONUNBUFFERED=1
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        - CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    
    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    image: whatcanyousee/verl:ngc-cu124-vllm0.8.5-sglang0.4.6-mcore0.12.0-te2.2
    commands:
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        - git clone https://github.com/volcengine/verl
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        - cd verl
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        - pip install --no-deps -e .
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        - pip install hf_transfer hf_xet
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        - |
        if [ $DSTACK_NODE_RANK = 0 ]; then
            python3 examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k
            python3 -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen2.5-7B-Instruct')" 
            ray start --head --port=6379;
        else
            ray start --address=$DSTACK_MASTER_NODE_IP:6379
        fi

    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # Expose Ray dashboard port
    ports:
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        - 8265

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    resources:
        gpu: 80GB:8
        shm_size: 128GB

    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # Save checkpoints on the instance
    volumes:
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        - /checkpoints:/checkpoints

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Now, if you run this task via `dstack apply`, it will automatically forward the Ray's dashboard port to `localhost:8265`.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code-block:: bash

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    dstack apply -f ray-cluster.dstack.yml

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
As long as the `dstack apply` is attached, you can use `localhost:8265` to submit Ray jobs for execution

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Submit Ray jobs
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
~~~~~~~~~~~~~~~

Before you can submit Ray jobs, ensure to install `ray` locally:
   
.. code-block:: shell

    pip install ray

Now you can submit the training job to the Ray cluster which is available at ``localhost:8265``:
   
.. code-block:: shell

    $ RAY_ADDRESS=http://localhost:8265
    $ ray job submit \
        -- python3 -m verl.trainer.main_ppo \
        data.train_files=/root/data/gsm8k/train.parquet \
        data.val_files=/root/data/gsm8k/test.parquet \
        data.train_batch_size=256 \
        data.max_prompt_length=512 \
        data.max_response_length=256 \
        actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
        critic.optim.lr=1e-5 \
        critic.model.path=Qwen/Qwen2.5-7B-Instruct \
        critic.ppo_micro_batch_size_per_gpu=4 \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.project_name=ppo_training \
        trainer.experiment_name=qwen-2.5-7B \
        trainer.val_before_train=False \
        trainer.default_hdfs_dir=null \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=2 \
        trainer.default_local_dir=/checkpoints \
        trainer.save_freq=10 \
        trainer.test_freq=10 \
        trainer.total_epochs=15 2>&1 | tee verl_demo.log \
        trainer.resume_mode=disable


For more details on how `dstack` works, check out its `documentation <https://dstack.ai/docs>`_.

How to debug?
---------------------


Ray Distributed Debugger VSCode Extension (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
1. Starting with Ray 2.39, Anyscale has introduced the `Ray Distributed Debugger <https://docs.ray.io/en/latest/ray-observability/ray-distributed-debugger.html>`_ VSCode extension. Follow the extension’s installation instructions, then add your cluster using the dashboard URL you obtained earlier.

   .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
   .. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/debugger.png?raw=true
      :alt: Ray Distributed Debugger VSCode extension screenshot

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
2. Prerequisites.

   .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
   Ensure the following are installed (see the extension README for more detail):

   .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
   - Visual Studio Code  
   .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
   - `ray[default]` >= 2.9.1  
   .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
   - `debugpy` >= 1.8.0  

   .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
   .. image:: https://github.com/aoshen524/verl/blob/main/docs/start/c7098b755ff689859837773a916c857.png?raw=true
      :alt: VSCode with Ray prerequisites

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
3. Environment Variables.

   .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
   To enable post‑mortem debugging, set:

   .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
   .. code-block:: bash

      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      export RAY_DEBUG_POST_MORTEM=1

   .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
   .. admonition:: Note
      :class: important

      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      Be sure to remove any legacy flags before starting Ray:

      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      - `RAY_DEBUG=legacy`  
      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      - `--ray-debugger-external`

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
4. Configuring BreakpointsSet up breakpoint() in your code, and submit job to cluster. Then the extension will show the breakpoint information.


   .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
   1. Insert `breakpoint()` calls into your remote functions.  
   .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
   2. Submit your job to the cluster.  

   .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
   The extension will detect active breakpoints and display them in VSCode.

   .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
   .. image:: https://github.com/aoshen524/verl/blob/main/docs/start/4ddad74395c79a1402331c0ce73316f.png?raw=true
      :alt: Detected breakpoint in VSCode

   .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
   **Note:** Breakpoints are only supported inside functions decorated with `@ray.remote`.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
5. Launching the Debugger.

   .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
   Run your job directly from the command line (do not use a `launch.json`):

   .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
   .. code-block:: bash

      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      python job.py

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
6. Attaching to a Breakpoint.

 .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
 Once the process hits the first `breakpoint()`, click the Ray Distributed Debugger icon in the VSCode sidebar to attach the debugger.

   .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
   .. image:: https://github.com/aoshen524/verl/blob/main/docs/start/4ddad74395c79a1402331c0ce73316f.png?raw=true
      :alt: Attaching VSCode debugger to Ray process

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
7. Debugging With Multiple breakpoint().

   .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
   For each subsequent task, first disconnect the current debugger session, then click the extension icon again to attach to the next breakpoint.

   .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
   .. image:: https://github.com/aoshen524/verl/blob/main/docs/start/6e83c910a62c82fecb89c6619e001cd.png?raw=true
      :alt: Disconnecting and reconnecting the debugger

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Legacy Ray Debugger
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
~~~~~~~~~~~~~~~~~~~
1. Ray has a builtin legacy `debugger <https://docs.ray.io/en/latest/ray-observability/user-guides/debug-apps/ray-debugging.html>`_ that allows you to debug your distributed applications. To enable debugger, start ray cluster with ``RAY_DEBUG=legacy`` and ``--ray-debugger-external``.

.. code-block:: bash

    # start head node
    RAY_DEBUG=legacy ray start --head --dashboard-host=0.0.0.0 --ray-debugger-external
    # start worker node
    RAY_DEBUG=legacy ray start --address='10.124.46.192:6379' --ray-debugger-external

2. Set up breakpoint in your code, and submit job to cluster. Then run ``ray debug`` to wait breakpoint:

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/legacy.png?raw=true


Multi-node training on AMD clusters
---------------------------------------------------------------------------------------

If you want to run multi-node training with slurm with Docker/Podman container on AMD Cluster, you can use the following script. 

If you encounter any issues in using AMD GPUs running verl, please contact `Yusheng Su <https://yushengsu-thu.github.io/>`_.

.. note::
    1. You need to use ``podman`` or ``docker`` in the following script. We will release the apptainer script later.
    2. If you want to use ``podman``, you just replace ``docker`` with ``podman`` in the following script.

The script includes the following steps:

1. SLURM Configuration
2. Environment Setup
3. Docker/Podman Container Setup
4. Ray Cluster Initialization
5. Data Preprocessing
6. Model Setup
7. Training Launch


slurm_script.sh
~~~~~~~~~~~~~~~~~~~~

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code-block:: bash

    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    #!/bin/bash

    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    #SBATCH --job-name=verl-ray-on-slurm
    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    #SBATCH --nodes=2
    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    #SBATCH --ntasks-per-node=2
    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    #SBATCH --mem=200G
    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    #SBATCH --time=30-00:00:00
    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    #SBATCH --gpus-per-node=8
    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    #SBATCH --cpus-per-task=28
    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    #SBATCH --output=../verl_log/slurm-%j.out
    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    #SBATCH --error=../verl_log/slurm-%j.err
    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    #SBATCH --nodelist=gpu-[0,1]


    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # load necessary modules
    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    ### Run this setup
    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # [Cluster]: Use docker
    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # docker pull docker.io/rocm/vllm:rocm6.2_mi300_ubuntu20.04_py3.9_vllm_0.6.4


    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    ##########################################################################
    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    ###The following setting should be set in different project and cluster###
    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    ##########################################################################

    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    ### Project
    CONTAINER_NAME="multinode_verl_training"
    IMG="verl.rocm"
    DOCKERFILE="docker/Dockerfile.rocm"
    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # echo $PWD
    verl_workdir="${HOME}/projects/verl_upstream"
    export TRANSFORMERS_CACHE="${HOME}/.cache/huggingface"
    export HF_HOME=$TRANSFORMERS_CACHE

    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    ### Cluster Network Setting
    export NCCL_DEBUG=TRACE
    export GPU_MAX_HW_QUEUES=2
    export TORCH_NCCL_HIGH_PRIORITY=1
    export NCCL_CHECKS_DISABLE=1
    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # export NCCL_IB_HCA=rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7 
    export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9
    export NCCL_IB_GID_INDEX=3
    export NCCL_CROSS_NIC=0
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export NCCL_PROTO=Simple
    export RCCL_MSCCL_ENABLE=0
    export TOKENIZERS_PARALLELISM=false
    export HSA_NO_SCRATCH_RECLAIM=1
    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    ##########################################################################

    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    ### For rocm and training script
    export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export ROCR_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES
    export CUDA_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES


    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # Build and launch the Docker container
    srun bash -c "
        .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
        # Exit on any error
        set -e 

        .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
        # Clean up dangling images (images with <none> tag)
        docker image prune -f

        .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
        # Need to pull the docker first
        docker pull docker.io/rocm/vllm:rocm6.2_mi300_ubuntu20.04_py3.9_vllm_0.6.4
        
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        if ! docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "${IMG}"; then
            echo \"Building ${IMG} image...\"
            docker build -f \"${DOCKERFILE}\" -t \"${IMG}\" .
        else
            echo \"${IMG} image already exists, skipping build\"
        fi

        .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
        # Removing old container if exists
        docker rm \"${CONTAINER_NAME}\" 2>/dev/null || true

        .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
        # Checking network devices
        ibdev2netdev

        .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
        # Launch the docker
        docker run --rm -d \
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        -e HYDRA_FULL_ERROR=1 \
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        -e HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES} \
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        -e ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES} \
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        -e CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        -e NCCL_DEBUG=${NCCL_DEBUG} \
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        -e GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES} \
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        -e TORCH_NCCL_HIGH_PRIORITY=${TORCH_NCCL_HIGH_PRIORITY} \
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        -e NCCL_CHECKS_DISABLE=${NCCL_CHECKS_DISABLE} \
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        -e NCCL_IB_HCA=${NCCL_IB_HCA} \
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        -e NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX} \
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        -e NCCL_CROSS_NIC=${NCCL_CROSS_NIC} \
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        -e CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS} \
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        -e NCCL_PROTO=${NCCL_PROTO} \
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        -e RCCL_MSCCL_ENABLE=${RCCL_MSCCL_ENABLE} \
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        -e TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM} \
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        -e HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM} \
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        -e TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE} \
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        -e HF_HOME=${HF_HOME} \
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        --network host \
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        --device /dev/dri \
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        --device /dev/kfd \
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        --device /dev/infiniband \
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        --group-add video \
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        --cap-add SYS_PTRACE \
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        --security-opt seccomp=unconfined \
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        --privileged \
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        -v \${HOME}:\${HOME} \
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        -v \${HOME}/.ssh:/root/.ssh \
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        -w "${verl_workdir}" \
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        --shm-size 128G \
        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        --name \"${CONTAINER_NAME}\" \
        \"${IMG}\" \
        tail -f /dev/null

        .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
        echo \"Container setup completed\"
    "
        .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
        # (Optional): If you do not want to root mode and require assign yuorself as the user
        .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
        # Please add `-e HOST_UID=$(id -u)` and `-e HOST_GID=$(id -g)` into the above docker launch script. 





    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    ### Ray launch the nodes before training

    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # Getting the node names
    nodes_array=($(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' '))

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    head_node=${nodes_array[0]}
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # if we detect a space character in the head node IP, we'll
    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # convert it to an ipv4 address. This step is optional.
    if [[ "$head_node_ip" == *" "* ]]; then
        IFS=' ' read -ra ADDR <<<"$head_node_ip"
    if [[ ${#ADDR[0]} -gt 16 ]]; then
        head_node_ip=${ADDR[1]}
    else
        head_node_ip=${ADDR[0]}
    fi
        echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
    fi

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    port=6379
    ip_head=$head_node_ip:$port
    export ip_head
    echo "IP Head: $ip_head"

    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # make sure we set environment variables before Ray initialization
    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # export VLLM_ATTENTION_BACKEND=XFORMERS

    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # Print out all env variables
    printenv

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    echo "Starting HEAD at $head_node"
    srun --nodes=1 --ntasks=1 -w "$head_node" \
        docker exec "${CONTAINER_NAME}" \
            ray start --head --node-ip-address="$head_node_ip" --port=$port \
            .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
            --dashboard-port=8266 \
            .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
            --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block &
    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # optional, though may be useful in certain versions of Ray < 1.0.
    sleep 10

    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # number of nodes other than the head node
    worker_num=$((SLURM_JOB_NUM_NODES - 1))

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    for ((i = 1; i <= worker_num; i++)); do
        node_i=${nodes_array[$i]}
        echo "Debug: Starting worker on node_i = ${node_i}"
        if [ -z "$node_i" ]; then
            echo "Error: Empty node name for worker $i"
            continue
        fi
        echo "Starting WORKER $i at $node_i"
        srun --nodes=1 --ntasks=1 -w "$node_i" \
            docker exec "${CONTAINER_NAME}" \
                ray start --address "$ip_head" --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block &
        sleep 5
    done




    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # Ray initlization test (See whether any error in the above excution)
    echo "Testing Ray initialization in the slurm nodes..."
    docker exec "${CONTAINER_NAME}" python3 -c '
    import ray
    try:
        ray.init(address="auto")
        print("\n=== Ray Cluster Status ===")
        print(f"Number of nodes: {len(ray.nodes())}")
        for node in ray.nodes():
            print("Node: {}, Status: {}".format(node["NodeManagerHostname"], node["Alive"]))
            .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
            # print(f"Node: {node}")
        ray.shutdown()
        print("Ray initialization successful!")
    except Exception as e:
        print(f"Ray initialization failed: {str(e)}")
    '
    echo "=== Ray test completed ==="
    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    ######



    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # Run data preprocessing

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    echo "Starting data preprocessing..."
    docker exec "${CONTAINER_NAME}" \
        python3 "examples/data_preprocess/gsm8k.py" "--local_dir" "../data/gsm8k"

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    echo "Starting data preprocessing..."
    docker exec "${CONTAINER_NAME}" \
        python3 "examples/data_preprocess/math_dataset.py" "--local_dir" "../data/math"

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    train_files="../data/gsm8k/train.parquet"
    val_files="../data/gsm8k/test.parquet"

    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # Download and test model
    echo "Loading model..."
    docker exec "${CONTAINER_NAME}" \
        python3 -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen2-7B-Instruct')"
    MODEL_PATH="Qwen/Qwen2-7B-Instruct"

    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # Set model path after pipeline test
    MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct"

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    echo "== Data and model loading Done =="

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    echo "Start to train..."

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    docker exec "${CONTAINER_NAME}" \
        python3 -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen2-7B-Instruct')"
    MODEL_PATH="Qwen/Qwen2-7B-Instruct"


    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    PYTHONUNBUFFERED=1 srun --overlap --nodes=${SLURM_NNODES} --ntasks=1 -w "$head_node" \
        docker exec "${CONTAINER_NAME}" \
        python3 -m verl.trainer.main_ppo \
        data.train_files=$train_files \
        data.val_files=$val_files \
        data.train_batch_size=1024 \
        data.max_prompt_length=1024 \
        data.max_response_length=1024 \
        actor_rollout_ref.model.path=$MODEL_PATH \
        actor_rollout_ref.model.enable_gradient_checkpointing=False \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=256 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        critic.optim.lr=1e-5 \
        critic.model.use_remove_padding=True \
        critic.model.path=$MODEL_PATH \
        critic.model.enable_gradient_checkpointing=False \
        critic.ppo_micro_batch_size_per_gpu=8 \
        critic.model.fsdp_config.param_offload=False \
        critic.model.fsdp_config.optimizer_offload=False \
        algorithm.kl_ctrl.kl_coef=0.0001 \
        trainer.critic_warmup=0 \
        trainer.logger=['console','wandb'] \
        trainer.project_name='verl_example' \
        trainer.experiment_name='Qwen2.5-32B-Instruct_function_rm' \
        trainer.n_gpus_per_node=${SLURM_GPUS_PER_NODE} \
        trainer.val_before_train=False \
        trainer.nnodes=${SLURM_NNODES} \
        trainer.save_freq=-1 \
        trainer.test_freq=10 \
        trainer.total_epochs=15


.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Run multi-node training with above slurm_script.sh
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
~~~~~~~~~~~~~~~~~~~~
Just sbatch your slurm_script.sh

.. code-block:: bash

    sbatch slurm_script.sh

