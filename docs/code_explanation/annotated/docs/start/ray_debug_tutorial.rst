.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Ray Debug Tutorial
==================

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. _wuxibin89: https://github.com/wuxibin89

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Author: `Ao Shen <https://aoshen524.github.io/>`_.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
How to debug?
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
---------------------


.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Ray Distributed Debugger VSCode Extension (Recommended)
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Starting with Ray 2.39, Anyscale has introduced the `Ray Distributed Debugger <https://docs.ray.io/en/latest/ray-observability/ray-distributed-debugger.html>`_ VSCode extension. Follow the extension’s installation instructions, then add your cluster using the dashboard URL you obtained earlier.

   .. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/debugger.png?raw=true
      :alt: Ray Distributed Debugger VSCode extension screenshot

2. Prerequisites.

   Ensure the following are installed (see the extension README for more detail):

   - Visual Studio Code  
   - `ray[default]` >= 2.9.1  
   - `debugpy` >= 1.8.0  

   .. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/readme.png?raw=true
      :alt: VSCode with Ray prerequisites

3. Environment Variables.

   To enable post‑mortem debugging, set:

   .. code-block:: bash

      export RAY_DEBUG_POST_MORTEM=1

   .. admonition:: Note
      :class: important

      Be sure to remove any legacy flags before starting Ray:

      - `RAY_DEBUG=legacy`  
      - `--ray-debugger-external`

4. Configuring BreakpointsSet up breakpoint() in your code, and submit job to cluster. Then the extension will show the breakpoint information.


   1. Insert `breakpoint()` calls into your remote functions.  
   2. Submit your job to the cluster.  

   The extension will detect active breakpoints and display them in VSCode.

   **Note:** Breakpoints are only supported inside functions decorated with `@ray.remote`.

5. Launching the Debugger.

   Run your job directly from the command line (do not use a `launch.json`):

   .. code-block:: bash

      python job.py

6. Attaching to a Breakpoint.

 Once the process hits the first `breakpoint()`, click the Ray Distributed Debugger icon in the VSCode sidebar to attach the debugger.

   .. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/launch.png?raw=true
      :alt: Attaching VSCode debugger to Ray process

7. Debugging With Multiple breakpoint().

   For each subsequent task, first disconnect the current debugger session, then click the extension icon again to attach to the next breakpoint.

   .. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/disconnect.png?raw=true
      :alt: Disconnecting and reconnecting the debugger

Legacy Ray Debugger
~~~~~~~~~~~~~~~~~~~
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
1. Ray has a builtin legacy `debugger <https://docs.ray.io/en/latest/ray-observability/user-guides/debug-apps/ray-debugging.html>`_ that allows you to debug your distributed applications. To enable debugger, start ray cluster with ``RAY_DEBUG=legacy`` and ``--ray-debugger-external``.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code-block:: bash

    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # start head node
    RAY_DEBUG=legacy ray start --head --dashboard-host=0.0.0.0 --ray-debugger-external
    .. [EXPLAIN] 以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。
    # start worker node
    RAY_DEBUG=legacy ray start --address='10.124.46.192:6379' --ray-debugger-external

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
2. Set up breakpoint in your code, and submit job to cluster. Then run ``ray debug`` to wait breakpoint:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/legacy.png?raw=true

