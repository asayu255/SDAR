.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Single Controller interface
============================

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
The Single Controller provides a unified interface for managing distributed workers
using Ray or other backends and executing functions across them.
It simplifies the process of dispatching tasks and collecting results, particularly 
when dealing with data parallelism or model parallelism. 


.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Core APIs
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
~~~~~~~~~~~~~~~~~

.. autoclass:: verl.single_controller.Worker
   :members: __init__, __new__, get_master_addr_port, get_cuda_visible_devices, world_size, rank

.. autoclass:: verl.single_controller.WorkerGroup
   :members: __init__,  world_size

.. autoclass:: verl.single_controller.ClassWithInitArgs
   :members: __init__, __call__

.. autoclass:: verl.single_controller.ResourcePool
   :members: __init__, world_size, local_world_size_list, local_rank_list

.. autoclass:: verl.single_controller.ray.RayWorkerGroup
   :members: __init__

.. autofunction:: verl.single_controller.ray.create_colocated_worker_cls