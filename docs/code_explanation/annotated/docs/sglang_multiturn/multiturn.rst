.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Multi-turn Rollout Support
==========================

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Basic Configuration
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
~~~~~~~~~~~~~~~~~~~

To enable multi-turn rollout, make sure to configure the following fields in your rollout configuration:

.. code-block:: yaml

    actor_rollout_ref: 
        rollout: 
            multi_turn: True
            name: "sglang"

These configuration activates the sglang engine for multi-turn interaction during rollout.

Custom Tool Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
For custom environment interaction tools, you can implement your own tools based on ``verl.tools.base_tool.BaseTool``. Then, specify your tool configurations in a YAML file:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code-block:: yaml

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    tools:
      .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
      - class_name: ""
        config: {}
        tool_schema:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
You may refer to GSM8KTool_example_configuration_, which is one example of the tool configurations. Its implementation can be found in gsm8k_tool.py_.

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
Finally, set the ``tools_config_file`` in your rollout config:

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. code-block:: yaml

    .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
    actor_rollout_ref:
        rollout:
            tool_kwargs:
                tools_config_file: <path_to_tool_yaml_file>

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
This allows integration of customized tool behaviors during actor rollout steps. 

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
GSM8K Multi-turn Training Performance  
.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See the training performance of multi-turn rollout on the GSM8K task HERE_.

.. _HERE: https://wandb.ai/zhaochenyang20/gsm8k_async_rl/runs/1ro1r7om?nw=nwuserzhaochenyang20

.. _GSM8KTool_example_configuration: https://github.com/volcengine/verl/blob/main/examples/sglang_multiturn/config/tool_config/gsm8k_tool_config.yaml

.. _gsm8k_tool.py: https://github.com/volcengine/verl/blob/main/verl/tools/gsm8k_tool.py

Search Tool Integration
~~~~~~~~~~~~~~~~~~~~~~~

.. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
.. toctree::
   :maxdepth: 1

   .. [EXPLAIN] この段落は実装の意図、利用条件または検証上の注意を説明する。
   search_tool_example