.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Frequently Asked Questions
====================================

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Ray related
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
------------

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
How to add breakpoint for debugging with distributed Ray?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Please checkout the official debugging guide from Ray: https://docs.ray.io/en/latest/ray-observability/ray-distributed-debugger.html


.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
"Unable to register worker with raylet"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
The cause of this issue is due to some system setting, e.g., SLURM added some constraints on how the CPUs are shared on a node. 
While `ray.init()` tries to launch as many worker processes as the number of CPU cores of the machine,
some constraints of SLURM restricts the `core-workers` seeing the `raylet` process, leading to the problem.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
To fix this issue, you can set the config term ``ray_init.num_cpus`` to a number allowed by your system.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Distributed training
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
------------------------

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
How to run multi-node post-training with Ray?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
You can start a ray cluster and submit a ray job, following the official guide from Ray: https://docs.ray.io/en/latest/ray-core/starting-ray.html

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Then in the configuration, set the ``trainer.nnode`` config to the number of machines for your job.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
How to use verl on a Slurm-managed cluster?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Ray provides users with `this <https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html>`_ official
tutorial to start a Ray cluster on top of Slurm. We have verified the :doc:`GSM8K example<../examples/gsm8k_example>`
on a Slurm cluster under a multi-node setting with the following steps.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
1. [Optional] If your cluster support `Apptainer or Singularity <https://apptainer.org/docs/user/main/>`_ and you wish
to use it, convert verl's Docker image to an Apptainer image. Alternatively, set up the environment with the package
manager available on your cluster or use other container runtimes (e.g. through `Slurm's OCI support <https://slurm.schedmd.com/containers.html>`_) available to you.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code:: bash

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    apptainer pull /your/dest/dir/vemlp-th2.4.0-cu124-vllm0.6.3-ray2.10-te1.7-v0.0.3.sif docker://verlai/verl:vemlp-th2.4.0-cu124-vllm0.6.3-ray2.10-te1.7-v0.0.3

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
2. Follow :doc:`GSM8K example<../examples/gsm8k_example>` to prepare the dataset and model checkpoints.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
3. Modify `examples/slurm/ray_on_slurm.slurm <https://github.com/volcengine/verl/blob/main/examples/slurm/ray_on_slurm.slurm>`_ with your cluster's own information.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
4. Submit the job script to the Slurm cluster with `sbatch`.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Please note that Slurm cluster setup may vary. If you encounter any issues, please refer to Ray's
`Slurm user guide <https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html>`_ for common caveats.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
If you changed Slurm resource specifications, please make sure to update the environment variables in the job script if necessary.


.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Install related
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
------------------------

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
NotImplementedError: TensorDict does not support membership checks with the `in` keyword. 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Detail error information: 

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code:: bash

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    NotImplementedError: TensorDict does not support membership checks with the `in` keyword. If you want to check if a particular key is in your TensorDict, please use `key in tensordict.keys()` instead.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Cause of the problem: There is no suitable version of tensordict package for the linux-arm64 platform. The confirmation method is as follows:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code:: bash

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    pip install tensordict==0.6.2

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Output example:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code:: bash

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    ERROR: Could not find a version that satisfies the requirement tensordict==0.6.2 (from versions: 0.0.1a0, 0.0.1b0, 0.0.1rc0, 0.0.2a0, 0.0.2b0, 0.0.3, 0.1.0, 0.1.1, 0.1.2, 0.8.0, 0.8.1, 0.8.2, 0.8.3)
    ERROR: No matching distribution found for tensordict==0.6.2

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Solution 1st:
  Install tensordict from source code:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code:: bash

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    pip uninstall tensordict
    git clone https://github.com/pytorch/tensordict.git
    cd tensordict/
    git checkout v0.6.2
    python setup.py develop
    pip install -v -e .

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Solution 2nd:
  Temperally modify the error takeplace codes: tensordict_var -> tensordict_var.keys()


.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Illegal memory access
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
---------------------------------

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
If you encounter the error message like ``CUDA error: an illegal memory access was encountered`` during rollout, most likely it is due to a known issue from vllm(<=0.6.3).
Please set the following environment variable. The env var must be set before the ``ray start`` command if any.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code:: bash

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    export VLLM_ATTENTION_BACKEND=XFORMERS

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
If in doubt, print this env var in each rank to make sure it is properly set.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Checkpoints
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
------------------------

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
If you want to convert the model checkpoint into huggingface safetensor format, please refer to ``scripts/model_merger.py``.


.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Triton ``compile_module_from_src`` error
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
------------------------------------------------

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
If you encounter triton compilation error similar to the stacktrace below, please set the ``use_torch_compile`` flag according to
https://verl.readthedocs.io/en/latest/examples/config.html to disable just-in-time compilation for fused kernels.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code:: bash

  .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
  File "/data/lbh/conda_envs/verl/lib/python3.10/site-packages/triton/runtime/jit.py", line 345, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
  File "/data/lbh/conda_envs/verl/lib/python3.10/site-packages/triton/runtime/autotuner.py", line 338, in run
    return self.fn.run(*args, **kwargs)
  File "/data/lbh/conda_envs/verl/lib/python3.10/site-packages/triton/runtime/jit.py", line 607, in run
    device = driver.active.get_current_device()
  File "/data/lbh/conda_envs/verl/lib/python3.10/site-packages/triton/runtime/driver.py", line 23, in __getattr__
    self._initialize_obj()
  File "/data/lbh/conda_envs/verl/lib/python3.10/site-packages/triton/runtime/driver.py", line 20, in _initialize_obj
    self._obj = self._init_fn()
  File "/data/lbh/conda_envs/verl/lib/python3.10/site-packages/triton/runtime/driver.py", line 9, in _create_driver
    return actives[0]()
  File "/data/lbh/conda_envs/verl/lib/python3.10/site-packages/triton/backends/nvidia/driver.py", line 371, in __init__
    self.utils = CudaUtils()  # TODO: make static
  File "/data/lbh/conda_envs/verl/lib/python3.10/site-packages/triton/backends/nvidia/driver.py", line 80, in __init__
    mod = compile_module_from_src(Path(os.path.join(dirname, "driver.c")).read_text(), "cuda_utils")
  File "/data/lbh/conda_envs/verl/lib/python3.10/site-packages/triton/backends/nvidia/driver.py", line 57, in compile_module_from_src
    so = _build(name, src_path, tmpdir, library_dirs(), include_dir, libraries)
  File "/data/lbh/conda_envs/verl/lib/python3.10/site-packages/triton/runtime/build.py", line 48, in _build
    ret = subprocess.check_call(cc_cmd)
  File "/data/lbh/conda_envs/verl/lib/python3.10/subprocess.py", line 369, in check_call
    raise CalledProcessError(retcode, cmd)

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
What is the meaning of train batch size, mini batch size, and micro batch size?
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
------------------------------------------------------------------------------------------

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
This figure illustrates the relationship between different batch size configurations.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
https://excalidraw.com/#json=pfhkRmiLm1jnnRli9VFhb,Ut4E8peALlgAUpr7E5pPCA

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. image:: https://github.com/user-attachments/assets/16aebad1-0da6-4eb3-806d-54a74e712c2d
