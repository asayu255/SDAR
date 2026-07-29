# 関数・メソッド索引

Python sourceのfunction/method定義。signatureは引数名だけを簡略表示する。

| function | kind / args | definition |
|---|---|---|
| `_agg_phase` | `def(rows, n_gpus)` | [`verl/utils/gpu_profiler.py`](../../verl/utils/gpu_profiler.py#L364) |
| `_agg_phase.mean` | `def(m)` | [`verl/utils/gpu_profiler.py`](../../verl/utils/gpu_profiler.py#L389) |
| `_as_dict` | `def(value)` | [`examples/data_preprocess/prepare_sdar_multitask.py`](../../examples/data_preprocess/prepare_sdar_multitask.py#L53) |
| `_Backend.sample` | `def(self)` | [`verl/utils/gpu_profiler.py`](../../verl/utils/gpu_profiler.py#L99) |
| `_bind_workers_method_to_parent` | `def(cls, key, user_defined_cls)` | [`verl/single_controller/ray/base.py`](../../verl/single_controller/ray/base.py#L626) |
| `_bind_workers_method_to_parent.generate_function` | `def(name, key)` | [`verl/single_controller/ray/base.py`](../../verl/single_controller/ray/base.py#L642) |
| `_bind_workers_method_to_parent.generate_function.async_func` | `async def(self, *args, **kwargs)` | [`verl/single_controller/ray/base.py`](../../verl/single_controller/ray/base.py#L647) |
| `_bind_workers_method_to_parent.generate_function.func` | `def(self, *args, **kwargs)` | [`verl/single_controller/ray/base.py`](../../verl/single_controller/ray/base.py#L643) |
| `_build_multitask_manager` | `def(config, tasks, task_max_steps, per_task_batch_size, group_n, is_train, seed, resources_per_worker)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L892) |
| `_build_split` | `def(search_dir, split, per_task_size_by_task, seed)` | [`examples/data_preprocess/prepare_sdar_multitask.py`](../../examples/data_preprocess/prepare_sdar_multitask.py#L113) |
| `_check_directory_structure` | `def(folder_path, record_file)` | [`verl/utils/fs.py`](../../verl/utils/fs.py#L174) |
| `_check_dispatch_mode` | `def(dispatch_mode)` | [`verl/single_controller/base/decorator.py`](../../verl/single_controller/base/decorator.py#L478) |
| `_check_execute_mode` | `def(execute_mode)` | [`verl/single_controller/base/decorator.py`](../../verl/single_controller/base/decorator.py#L486) |
| `_compute_mlflow_params_from_objects` | `def(params)` | [`verl/utils/tracking.py`](../../verl/utils/tracking.py#L222) |
| `_compute_response_info` | `def(batch)` | [`recipe/spin/spin_trainer.py`](../../recipe/spin/spin_trainer.py#L140) |
| `_compute_response_info` | `def(batch)` | [`verl/trainer/ppo/metric_utils.py`](../../verl/trainer/ppo/metric_utils.py#L165) |
| `_concat_data_proto_or_future` | `def(output)` | [`verl/single_controller/base/decorator.py`](../../verl/single_controller/base/decorator.py#L207) |
| `_config_from_expectations` | `def(expect_file)` | [`tests/trainer/test_expected_config.py`](../../tests/trainer/test_expected_config.py#L44) |
| `_configure_default_logger` | `def()` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L423) |
| `_copy` | `def(from_path, to_path, timeout)` | [`verl/utils/hdfs_io.py`](../../verl/utils/hdfs_io.py#L113) |
| `_copy_config_for_task` | `def(config, env_name, max_steps, task)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L883) |
| `_custom_flash_attention_forward` | `def(query_states, key_states, value_states, attention_mask, query_length, is_causal, position_ids, sliding_window, use_top_left_mask, deterministic, **kwargs)` | [`verl/models/transformers/qwen2_vl.py`](../../verl/models/transformers/qwen2_vl.py#L182) |
| `_DataProtoConfigMeta.auto_padding` | `def(cls)` | [`verl/protocol.py`](../../verl/protocol.py#L53) |
| `_DataProtoConfigMeta.auto_padding` | `def(cls, enabled)` | [`verl/protocol.py`](../../verl/protocol.py#L58) |
| `_default_compute_score` | `def(data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore)` | [`verl/utils/reward_score/__init__.py`](../../verl/utils/reward_score/__init__.py#L98) |
| `_demo` | `def()` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L456) |
| `_determine_fsdp_megatron_base_class` | `def(mros)` | [`verl/single_controller/ray/base.py`](../../verl/single_controller/ray/base.py#L678) |
| `_drop_batch_level_metrics` | `def(metrics)` | [`verl/trainer/ppo/metric_utils.py`](../../verl/trainer/ppo/metric_utils.py#L146) |
| `_dummy_task_dataframe` | `def(task_name, split, size)` | [`examples/data_preprocess/prepare_sdar_multitask.py`](../../examples/data_preprocess/prepare_sdar_multitask.py#L31) |
| `_ensure_sampler` | `def()` | [`verl/utils/gpu_profiler.py`](../../verl/utils/gpu_profiler.py#L306) |
| `_env_flag` | `def(name, default)` | [`verl/utils/gpu_profiler.py`](../../verl/utils/gpu_profiler.py#L67) |
| `_env_kwargs_equal` | `def(a, b)` | [`agent_system/multi_turn_rollout/rollout_loop.py`](../../agent_system/multi_turn_rollout/rollout_loop.py#L109) |
| `_exists` | `def(file_path)` | [`verl/utils/hdfs_io.py`](../../verl/utils/hdfs_io.py#L43) |
| `_expand_mask` | `def(mask, dtype, tgt_len)` | [`verl/models/llama/megatron/modeling_llama_megatron.py`](../../verl/models/llama/megatron/modeling_llama_megatron.py#L60) |
| `_expand_mask` | `def(mask, dtype, tgt_len)` | [`verl/models/qwen2/megatron/modeling_qwen2_megatron.py`](../../verl/models/qwen2/megatron/modeling_qwen2_megatron.py#L60) |
| `_expand_mask` | `def(mask, dtype, tgt_len)` | [`verl/utils/torch_functional.py`](../../verl/utils/torch_functional.py#L563) |
| `_FakeBackend.__init__` | `def(self, n)` | [`verl/utils/gpu_profiler.py`](../../verl/utils/gpu_profiler.py#L482) |
| `_FakeBackend.sample` | `def(self)` | [`verl/utils/gpu_profiler.py`](../../verl/utils/gpu_profiler.py#L485) |
| `_FakeTeacherWG.__init__` | `def(self, code)` | [`tests/trainer/test_opd_routing.py`](../../tests/trainer/test_opd_routing.py#L38) |
| `_FakeTeacherWG.compute_ref_log_prob` | `def(self, sub)` | [`tests/trainer/test_opd_routing.py`](../../tests/trainer/test_opd_routing.py#L42) |
| `_fix_a_slash_b` | `def(string)` | [`verl/utils/reward_score/prime_math/math_normalize.py`](../../verl/utils/reward_score/prime_math/math_normalize.py#L90) |
| `_fix_fracs` | `def(string)` | [`verl/utils/reward_score/prime_math/math_normalize.py`](../../verl/utils/reward_score/prime_math/math_normalize.py#L58) |
| `_fix_sqrt` | `def(string)` | [`verl/utils/reward_score/prime_math/math_normalize.py`](../../verl/utils/reward_score/prime_math/math_normalize.py#L115) |
| `_flatten_dict` | `def(raw)` | [`verl/utils/tracking.py`](../../verl/utils/tracking.py#L249) |
| `_fmt` | `def(v, spec)` | [`verl/utils/gpu_profiler.py`](../../verl/utils/gpu_profiler.py#L424) |
| `_fmt_per_gpu` | `def(vals)` | [`agent_system/multi_turn_rollout/rollout_loop.py`](../../agent_system/multi_turn_rollout/rollout_loop.py#L136) |
| `_fsdp_activation_offloading_test` | `def(rank, world_size, rendezvous_file, strategy)` | [`tests/utils/gpu_tests/test_activation_offload.py`](../../tests/utils/gpu_tests/test_activation_offload.py#L32) |
| `_fused_linear_for_ppo_bwd` | `def(dlog_probs, dentropy, hidden_states, vocab_weights, input_ids, temperature)` | [`verl/utils/experimental/torch_functional.py`](../../verl/utils/experimental/torch_functional.py#L39) |
| `_fused_linear_for_ppo_fwd` | `def(hidden_states, vocab_weights, input_ids, temperature)` | [`verl/utils/experimental/torch_functional.py`](../../verl/utils/experimental/torch_functional.py#L19) |
| `_get_base_transformer_config` | `def(hf_config, dtype, **override_transformer_config_kwargs)` | [`verl/models/mcore/config_converter.py`](../../verl/models/mcore/config_converter.py#L32) |
| `_get_cpu_memory` | `def()` | [`scripts/diagnose.py`](../../scripts/diagnose.py#L208) |
| `_get_current_git_commit` | `def()` | [`scripts/diagnose.py`](../../scripts/diagnose.py#L89) |
| `_get_current_mem_info` | `def(unit, precision)` | [`verl/utils/debug/performance.py`](../../verl/utils/debug/performance.py#L31) |
| `_get_free_port` | `def()` | [`tests/workers/rollout/test_vllm_tool_calling.py`](../../tests/workers/rollout/test_vllm_tool_calling.py#L39) |
| `_get_free_port` | `def()` | [`verl/workers/rollout/async_server.py`](../../verl/workers/rollout/async_server.py#L44) |
| `_get_gpu_info` | `def()` | [`scripts/diagnose.py`](../../scripts/diagnose.py#L216) |
| `_get_input_embeds` | `def(model, input_ids, attention_mask, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw)` | [`verl/models/transformers/qwen2_vl.py`](../../verl/models/transformers/qwen2_vl.py#L335) |
| `_get_input_embeds` | `def(model, input_ids, attention_mask, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw)` | [`verl/models/transformers/qwen3_vl.py`](../../verl/models/transformers/qwen3_vl.py#L136) |
| `_get_invalid_action_penalty_coef` | `def(data_item, invalid_action_penalty_coef, invalid_action_penalty_coef_by_task)` | [`verl/trainer/ppo/ray_trainer.py`](../../verl/trainer/ppo/ray_trainer.py#L208) |
| `_get_mla_transformer_config` | `def(hf_config, mla_rope_config, dtype, **override_transformer_config_kwargs)` | [`verl/models/mcore/config_converter.py`](../../verl/models/mcore/config_converter.py#L99) |
| `_get_multitask_per_task_batch_size` | `def(config, tasks, is_train)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L856) |
| `_get_multitask_task_history_length` | `def(config, task)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L875) |
| `_get_multitask_task_max_steps` | `def(config, tasks)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L847) |
| `_get_multitask_tasks` | `def(config)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L841) |
| `_get_parallel_model_architecture_from_config` | `def(config, value)` | [`verl/utils/model.py`](../../verl/utils/model.py#L267) |
| `_get_qualified_name` | `def(func)` | [`verl/utils/import_utils.py`](../../verl/utils/import_utils.py#L117) |
| `_get_system_info` | `def()` | [`scripts/diagnose.py`](../../scripts/diagnose.py#L244) |
| `_get_unique_tensor_key` | `def(tensor)` | [`verl/utils/activation_offload.py`](../../verl/utils/activation_offload.py#L34) |
| `_hdfs_cmd` | `def(cmd)` | [`verl/utils/hdfs_io.py`](../../verl/utils/hdfs_io.py#L144) |
| `_hf_casual_fwd` | `def(config, sp_size, dp_size)` | [`tests/models/test_transformers_ulysses.py`](../../tests/models/test_transformers_ulysses.py#L86) |
| `_hf_casual_fwd_bwd` | `def(config, sp_size, dp_size)` | [`tests/models/test_transformers_ulysses.py`](../../tests/models/test_transformers_ulysses.py#L147) |
| `_init_args` | `def()` | [`scripts/converter_hf_to_mcore.py`](../../scripts/converter_hf_to_mcore.py#L32) |
| `_inject_implicit_mixed_number` | `def(step)` | [`verl/utils/reward_score/prime_math/__init__.py`](../../verl/utils/reward_score/prime_math/__init__.py#L105) |
| `_is_float` | `def(num)` | [`verl/utils/reward_score/prime_math/__init__.py`](../../verl/utils/reward_score/prime_math/__init__.py#L71) |
| `_is_frac` | `def(expr)` | [`verl/utils/reward_score/prime_math/__init__.py`](../../verl/utils/reward_score/prime_math/__init__.py#L86) |
| `_is_int` | `def(x)` | [`verl/utils/reward_score/prime_math/__init__.py`](../../verl/utils/reward_score/prime_math/__init__.py#L79) |
| `_is_non_local` | `def(path)` | [`verl/utils/hdfs_io.py`](../../verl/utils/hdfs_io.py#L148) |
| `_last_boxed_only_string` | `def(string)` | [`verl/utils/reward_score/prime_math/__init__.py`](../../verl/utils/reward_score/prime_math/__init__.py#L307) |
| `_leaf_envs` | `def(env_manager)` | [`agent_system/environments/resume.py`](../../agent_system/environments/resume.py#L44) |
| `_left_pad_tensor` | `def(tensor, target_len, pad_value)` | [`verl/utils/dataset/rl_dataset.py`](../../verl/utils/dataset/rl_dataset.py#L37) |
| `_load_hf_model` | `def(config, model_config, is_value_model, local_cache_path)` | [`verl/utils/model.py`](../../verl/utils/model.py#L277) |
| `_logsoftmax_at` | `def(logits, ids)` | [`tests/trainer/test_opd_routing.py`](../../tests/trainer/test_opd_routing.py#L152) |
| `_make_backend` | `def()` | [`verl/utils/gpu_profiler.py`](../../verl/utils/gpu_profiler.py#L192) |
| `_make_batch` | `def(num_rows, seq_len, response_length, seed)` | [`tests/ray_cpu/test_rollout_speedup_mechanisms.py`](../../tests/ray_cpu/test_rollout_speedup_mechanisms.py#L82) |
| `_make_batch` | `def(task_names, batch_size, traj_uids, response_lengths)` | [`tests/trainer/ppo/test_per_task_metrics.py`](../../tests/trainer/ppo/test_per_task_metrics.py#L37) |
| `_make_batch` | `def(task_names, resp_len)` | [`tests/trainer/test_opd_routing.py`](../../tests/trainer/test_opd_routing.py#L51) |
| `_make_causal_mask` | `def(input_ids_shape, dtype, device)` | [`verl/models/llama/megatron/modeling_llama_megatron.py`](../../verl/models/llama/megatron/modeling_llama_megatron.py#L47) |
| `_make_causal_mask` | `def(input_ids_shape, dtype, device)` | [`verl/models/qwen2/megatron/modeling_qwen2_megatron.py`](../../verl/models/qwen2/megatron/modeling_qwen2_megatron.py#L47) |
| `_make_causal_mask` | `def(input_ids_shape, dtype, device)` | [`verl/utils/torch_functional.py`](../../verl/utils/torch_functional.py#L550) |
| `_make_trainer` | `def(teacher_wg)` | [`tests/trainer/test_opd_routing.py`](../../tests/trainer/test_opd_routing.py#L62) |
| `_materialize_futures` | `def(*args, **kwargs)` | [`verl/single_controller/base/decorator.py`](../../verl/single_controller/base/decorator.py#L490) |
| `_megatron_calc_global_rank` | `def(tp_rank, dp_rank, pp_rank)` | [`verl/models/llama/megatron/checkpoint_utils/llama_saver.py`](../../verl/models/llama/megatron/checkpoint_utils/llama_saver.py#L27) |
| `_megatron_calc_global_rank` | `def(tp_rank, dp_rank, pp_rank, cp_rank, ep_rank)` | [`verl/models/mcore/saver.py`](../../verl/models/mcore/saver.py#L28) |
| `_megatron_calc_global_rank` | `def(tp_rank, dp_rank, pp_rank)` | [`verl/models/qwen2/megatron/checkpoint_utils/qwen2_saver.py`](../../verl/models/qwen2/megatron/checkpoint_utils/qwen2_saver.py#L27) |
| `_megatron_calc_layer_map` | `def(config)` | [`verl/models/llama/megatron/checkpoint_utils/llama_loader.py`](../../verl/models/llama/megatron/checkpoint_utils/llama_loader.py#L21) |
| `_megatron_calc_layer_map` | `def(config)` | [`verl/models/llama/megatron/checkpoint_utils/llama_loader_depracated.py`](../../verl/models/llama/megatron/checkpoint_utils/llama_loader_depracated.py#L21) |
| `_megatron_calc_layer_map` | `def(config)` | [`verl/models/llama/megatron/checkpoint_utils/llama_saver.py`](../../verl/models/llama/megatron/checkpoint_utils/llama_saver.py#L38) |
| `_megatron_calc_layer_map` | `def(config)` | [`verl/models/mcore/loader.py`](../../verl/models/mcore/loader.py#L24) |
| `_megatron_calc_layer_map` | `def(config)` | [`verl/models/mcore/saver.py`](../../verl/models/mcore/saver.py#L47) |
| `_megatron_calc_layer_map` | `def(config)` | [`verl/models/qwen2/megatron/checkpoint_utils/qwen2_loader.py`](../../verl/models/qwen2/megatron/checkpoint_utils/qwen2_loader.py#L21) |
| `_megatron_calc_layer_map` | `def(config)` | [`verl/models/qwen2/megatron/checkpoint_utils/qwen2_loader_depracated.py`](../../verl/models/qwen2/megatron/checkpoint_utils/qwen2_loader_depracated.py#L21) |
| `_megatron_calc_layer_map` | `def(config)` | [`verl/models/qwen2/megatron/checkpoint_utils/qwen2_saver.py`](../../verl/models/qwen2/megatron/checkpoint_utils/qwen2_saver.py#L38) |
| `_merge_with_image_features` | `def(self, inputs_embeds, input_ids, image_features)` | [`verl/models/transformers/kimi_vl.py`](../../verl/models/transformers/kimi_vl.py#L25) |
| `_mkdir` | `def(file_path)` | [`verl/utils/hdfs_io.py`](../../verl/utils/hdfs_io.py#L75) |
| `_MlflowLoggingAdapter.log` | `def(self, data, step)` | [`verl/utils/tracking.py`](../../verl/utils/tracking.py#L215) |
| `_mock_api_call_for_concurrency_tracking` | `def(active_calls_counter, max_calls_tracker, call_lock, sandbox_fusion_url, code, stdin, compile_timeout, run_timeout, language)` | [`tests/reward_score/test_sandbox_fusion.py`](../../tests/reward_score/test_sandbox_fusion.py#L324) |
| `_mp_target_wrapper` | `def(target_func, mp_queue, args, kwargs)` | [`verl/utils/py_functional.py`](../../verl/utils/py_functional.py#L30) |
| `_normalize` | `def(value)` | [`verl/utils/expected_config.py`](../../verl/utils/expected_config.py#L46) |
| `_normalize` | `def(expr)` | [`verl/utils/reward_score/prime_math/__init__.py`](../../verl/utils/reward_score/prime_math/__init__.py#L126) |
| `_normalize_multitask_name` | `def(task_name)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L604) |
| `_normalize_task_name` | `def(task_name)` | [`verl/trainer/ppo/rlsd_utils.py`](../../verl/trainer/ppo/rlsd_utils.py#L34) |
| `_now` | `def()` | [`agent_system/multi_turn_rollout/rollout_loop.py`](../../agent_system/multi_turn_rollout/rollout_loop.py#L132) |
| `_NvmlBackend.__init__` | `def(self)` | [`verl/utils/gpu_profiler.py`](../../verl/utils/gpu_profiler.py#L104) |
| `_NvmlBackend._safe` | `def(self, fn, *a)` | [`verl/utils/gpu_profiler.py`](../../verl/utils/gpu_profiler.py#L112) |
| `_NvmlBackend.sample` | `def(self)` | [`verl/utils/gpu_profiler.py`](../../verl/utils/gpu_profiler.py#L118) |
| `_pad_tensor` | `def(x, dim, padding_size)` | [`verl/utils/ulysses.py`](../../verl/utils/ulysses.py#L104) |
| `_parse_latex` | `def(expr)` | [`verl/utils/reward_score/prime_math/__init__.py`](../../verl/utils/reward_score/prime_math/__init__.py#L53) |
| `_passages2string` | `def(retrieval_result)` | [`agent_system/environments/env_package/search/third_party/skyrl_gym/tools/search.py`](../../agent_system/environments/env_package/search/third_party/skyrl_gym/tools/search.py#L155) |
| `_passages2string` | `def(retrieval_result)` | [`verl/tools/utils/search_r1_like_utils.py`](../../verl/tools/utils/search_r1_like_utils.py#L132) |
| `_plain_container` | `def(value)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L615) |
| `_post_process_outputs` | `def(tokenizer, output)` | [`verl/workers/rollout/sglang_rollout/sglang_rollout.py`](../../verl/workers/rollout/sglang_rollout/sglang_rollout.py#L88) |
| `_post_process_outputs._map_each_response` | `def(resp)` | [`verl/workers/rollout/sglang_rollout/sglang_rollout.py`](../../verl/workers/rollout/sglang_rollout/sglang_rollout.py#L89) |
| `_postprocess_action` | `def(action)` | [`agent_system/environments/env_package/search/projection.py`](../../agent_system/environments/env_package/search/projection.py#L20) |
| `_pre_process_inputs` | `def(pad_token_id, prompt_token_ids)` | [`tests/workers/rollout/test_sglang_spmd.py`](../../tests/workers/rollout/test_sglang_spmd.py#L37) |
| `_pre_process_inputs` | `def(pad_token_id, prompt_token_ids)` | [`verl/workers/rollout/sglang_rollout/sglang_rollout.py`](../../verl/workers/rollout/sglang_rollout/sglang_rollout.py#L77) |
| `_pre_process_inputs` | `def(pad_token_id, prompt_token_ids)` | [`verl/workers/rollout/vllm_rollout/fire_vllm_rollout.py`](../../verl/workers/rollout/vllm_rollout/fire_vllm_rollout.py#L49) |
| `_pre_process_inputs` | `def(pad_token_id, prompt_token_ids)` | [`verl/workers/rollout/vllm_rollout/vllm_rollout.py`](../../verl/workers/rollout/vllm_rollout/vllm_rollout.py#L59) |
| `_pre_process_inputs` | `def(pad_token_id, prompt_token_ids)` | [`verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`](../../verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py#L78) |
| `_preprocess_tensor_for_update_weights` | `def(tensor)` | [`verl/workers/sharding_manager/fsdp_sglang.py`](../../verl/workers/sharding_manager/fsdp_sglang.py#L43) |
| `_print_report` | `def(samples, n_gpus, label, interval)` | [`verl/utils/gpu_profiler.py`](../../verl/utils/gpu_profiler.py#L428) |
| `_print_turn_timing` | `def(records)` | [`agent_system/multi_turn_rollout/rollout_loop.py`](../../agent_system/multi_turn_rollout/rollout_loop.py#L142) |
| `_process_pool_worker_for_concurrency_test` | `def(sandbox_url, in_outs, generation, language, timeout, mp_semaphore_for_check_correctness, active_calls_counter, max_calls_tracker, call_lock)` | [`tests/reward_score/test_sandbox_fusion.py`](../../tests/reward_score/test_sandbox_fusion.py#L358) |
| `_process_search_row` | `def(row, split, idx)` | [`examples/data_preprocess/prepare_sdar_multitask.py`](../../examples/data_preprocess/prepare_sdar_multitask.py#L57) |
| `_process_single_case` | `def(case_index, stdin_data, expected_output, sandbox_fusion_url, generation, timeout, language, concurrent_semaphore, fn_name)` | [`verl/utils/reward_score/sandbox_fusion/utils.py`](../../verl/utils/reward_score/sandbox_fusion/utils.py#L128) |
| `_record_directory_structure` | `def(folder_path)` | [`verl/utils/fs.py`](../../verl/utils/fs.py#L160) |
| `_remove_right_units` | `def(string)` | [`verl/utils/reward_score/prime_math/math_normalize.py`](../../verl/utils/reward_score/prime_math/math_normalize.py#L105) |
| `_repeat_interleave` | `def(value, repeats)` | [`verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`](../../verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py#L87) |
| `_rolled_labels` | `def(labels, input_ids)` | [`verl/models/transformers/dense_common.py`](../../verl/models/transformers/dense_common.py#L43) |
| `_run_case` | `def(n_traj, max_steps, rollout_n, done_at, slow_set, max_in_flight)` | [`tests/ray_cpu/test_async_rollout_equivalence.py`](../../tests/ray_cpu/test_async_rollout_equivalence.py#L134) |
| `_run_case.env_fn` | `async def(i, step, action)` | [`tests/ray_cpu/test_async_rollout_equivalence.py`](../../tests/ray_cpu/test_async_rollout_equivalence.py#L155) |
| `_run_case.gen_fn` | `async def(i, step, obs)` | [`tests/ray_cpu/test_async_rollout_equivalence.py`](../../tests/ray_cpu/test_async_rollout_equivalence.py#L151) |
| `_run_case.traj_uid_factory` | `def()` | [`tests/ray_cpu/test_async_rollout_equivalence.py`](../../tests/ray_cpu/test_async_rollout_equivalence.py#L147) |
| `_run_case.uid_factory` | `def()` | [`tests/ray_cpu/test_async_rollout_equivalence.py`](../../tests/ray_cpu/test_async_rollout_equivalence.py#L143) |
| `_run_cmd` | `def(cmd, timeout)` | [`verl/utils/hdfs_io.py`](../../verl/utils/hdfs_io.py#L140) |
| `_sample_dataframe` | `def(df, size, seed)` | [`examples/data_preprocess/prepare_sdar_multitask.py`](../../examples/data_preprocess/prepare_sdar_multitask.py#L18) |
| `_Sampler.__init__` | `def(self, backend, interval)` | [`verl/utils/gpu_profiler.py`](../../verl/utils/gpu_profiler.py#L211) |
| `_Sampler._current_phase` | `def(self)` | [`verl/utils/gpu_profiler.py`](../../verl/utils/gpu_profiler.py#L241) |
| `_Sampler._run` | `def(self)` | [`verl/utils/gpu_profiler.py`](../../verl/utils/gpu_profiler.py#L245) |
| `_Sampler.mean_util_between` | `def(self, t0, t1)` | [`verl/utils/gpu_profiler.py`](../../verl/utils/gpu_profiler.py#L262) |
| `_Sampler.per_gpu_util_between` | `def(self, t0, t1)` | [`verl/utils/gpu_profiler.py`](../../verl/utils/gpu_profiler.py#L271) |
| `_Sampler.pop` | `def(self, name)` | [`verl/utils/gpu_profiler.py`](../../verl/utils/gpu_profiler.py#L231) |
| `_Sampler.push` | `def(self, name)` | [`verl/utils/gpu_profiler.py`](../../verl/utils/gpu_profiler.py#L227) |
| `_Sampler.report_and_reset` | `def(self, label)` | [`verl/utils/gpu_profiler.py`](../../verl/utils/gpu_profiler.py#L288) |
| `_Sampler.stop` | `def(self)` | [`verl/utils/gpu_profiler.py`](../../verl/utils/gpu_profiler.py#L297) |
| `_search_dataframe` | `def(search_dir, split, size, seed)` | [`examples/data_preprocess/prepare_sdar_multitask.py`](../../examples/data_preprocess/prepare_sdar_multitask.py#L89) |
| `_select_rm_score_fn` | `def(data_source)` | [`examples/split_placement/main_ppo_split.py`](../../examples/split_placement/main_ppo_split.py#L28) |
| `_SmiBackend.__init__` | `def(self)` | [`verl/utils/gpu_profiler.py`](../../verl/utils/gpu_profiler.py#L145) |
| `_SmiBackend._f` | `def(tok)` | [`verl/utils/gpu_profiler.py`](../../verl/utils/gpu_profiler.py#L164) |
| `_SmiBackend._raw` | `def(self)` | [`verl/utils/gpu_profiler.py`](../../verl/utils/gpu_profiler.py#L154) |
| `_SmiBackend.sample` | `def(self)` | [`verl/utils/gpu_profiler.py`](../../verl/utils/gpu_profiler.py#L171) |
| `_split_args_kwargs_data_proto` | `def(chunks, *args, **kwargs)` | [`verl/single_controller/base/decorator.py`](../../verl/single_controller/base/decorator.py#L77) |
| `_split_args_kwargs_data_proto_with_auto_padding` | `def(chunks, *args, **kwargs)` | [`verl/single_controller/base/decorator.py`](../../verl/single_controller/base/decorator.py#L93) |
| `_str_is_int` | `def(x)` | [`verl/utils/reward_score/prime_math/__init__.py`](../../verl/utils/reward_score/prime_math/__init__.py#L90) |
| `_str_to_int` | `def(x)` | [`verl/utils/reward_score/prime_math/__init__.py`](../../verl/utils/reward_score/prime_math/__init__.py#L99) |
| `_strip_properly_formatted_commas` | `def(expr)` | [`verl/utils/reward_score/prime_math/__init__.py`](../../verl/utils/reward_score/prime_math/__init__.py#L115) |
| `_strip_string` | `def(string)` | [`verl/utils/reward_score/prime_math/math_normalize.py`](../../verl/utils/reward_score/prime_math/math_normalize.py#L130) |
| `_sympy_parse` | `def(expr)` | [`verl/utils/reward_score/prime_math/__init__.py`](../../verl/utils/reward_score/prime_math/__init__.py#L44) |
| `_temp_run` | `def(in_outs, generation, debug, result, metadata_list, timeout)` | [`recipe/r1/tasks/livecodebench.py`](../../recipe/r1/tasks/livecodebench.py#L25) |
| `_temp_run` | `def(sample, generation, debug, result, metadata_list, timeout)` | [`verl/utils/reward_score/prime_code/utils.py`](../../verl/utils/reward_score/prime_code/utils.py#L26) |
| `_TensorboardAdapter.__init__` | `def(self)` | [`verl/utils/tracking.py`](../../verl/utils/tracking.py#L196) |
| `_TensorboardAdapter.finish` | `def(self)` | [`verl/utils/tracking.py`](../../verl/utils/tracking.py#L210) |
| `_TensorboardAdapter.log` | `def(self, data, step)` | [`verl/utils/tracking.py`](../../verl/utils/tracking.py#L206) |
| `_timer` | `def(name, timing_raw)` | [`recipe/hgpo/hgpo_ray_trainer.py`](../../recipe/hgpo/hgpo_ray_trainer.py#L386) |
| `_timer` | `def(name, timing_raw)` | [`recipe/spin/spin_trainer.py`](../../recipe/spin/spin_trainer.py#L343) |
| `_timer` | `def(name, timing_raw)` | [`verl/trainer/ppo/ray_trainer.py`](../../verl/trainer/ppo/ray_trainer.py#L405) |
| `_to_plain_dict` | `def(value)` | [`verl/trainer/ppo/rlsd_utils.py`](../../verl/trainer/ppo/rlsd_utils.py#L47) |
| `_transform_params_to_json_serializable` | `def(x, convert_list_to_dict)` | [`verl/utils/tracking.py`](../../verl/utils/tracking.py#L229) |
| `_ulysses_flash_attention_forward` | `def(query_states, key_states, value_states, attention_mask, query_length, *args, **kwargs)` | [`verl/models/transformers/monkey_patch.py`](../../verl/models/transformers/monkey_patch.py#L49) |
| `_ulysses_flash_attn_forward` | `def(self, hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, **kwargs)` | [`verl/models/transformers/kimi_vl.py`](../../verl/models/transformers/kimi_vl.py#L126) |
| `_unpad_tensor` | `def(x, dim, padding_size)` | [`verl/utils/ulysses.py`](../../verl/utils/ulysses.py#L111) |
| `_unwrap_ray_remote` | `def(cls)` | [`verl/single_controller/ray/base.py`](../../verl/single_controller/ray/base.py#L672) |
| `_values_equal` | `def(got, want)` | [`verl/utils/expected_config.py`](../../verl/utils/expected_config.py#L66) |
| `_VocabParallelEntropy.backward` | `def(ctx, grad_output)` | [`verl/utils/megatron/tensor_parallel.py`](../../verl/utils/megatron/tensor_parallel.py#L130) |
| `_VocabParallelEntropy.forward` | `def(ctx, vocab_parallel_logits)` | [`verl/utils/megatron/tensor_parallel.py`](../../verl/utils/megatron/tensor_parallel.py#L111) |
| `_VocabParallelEntropy.forward.mul_reduce` | `def(a, b)` | [`verl/utils/megatron/tensor_parallel.py`](../../verl/utils/megatron/tensor_parallel.py#L113) |
| `_worker` | `def(rank, world_size, init_method, max_token_len, use_same_dp, min_mb)` | [`tests/utils/gpu_tests/test_seqlen_balancing.py`](../../tests/utils/gpu_tests/test_seqlen_balancing.py#L41) |
| `_worker_mask` | `def(rank, world_size, rendezvous_file)` | [`tests/utils/gpu_tests/test_torch_functional.py`](../../tests/utils/gpu_tests/test_torch_functional.py#L68) |
| `_worker_mean` | `def(rank, world_size, rendezvous_file)` | [`tests/utils/gpu_tests/test_torch_functional.py`](../../tests/utils/gpu_tests/test_torch_functional.py#L25) |
| `acc_reward` | `def(predict_str, ground_truth)` | [`verl/utils/reward_score/geo3k.py`](../../verl/utils/reward_score/geo3k.py#L26) |
| `ActivationHandler.__init__` | `def(self, offload_ctx, sync_func, tensor_filter, enable_ckpt)` | [`verl/utils/activation_offload.py`](../../verl/utils/activation_offload.py#L411) |
| `ActivationHandler._ckpt_forward` | `def(self, forward_method, *args, **kwargs)` | [`verl/utils/activation_offload.py`](../../verl/utils/activation_offload.py#L448) |
| `ActivationHandler._ckpt_forward.my_function` | `def(*inputs)` | [`verl/utils/activation_offload.py`](../../verl/utils/activation_offload.py#L451) |
| `ActivationHandler._pack_kwargs` | `def(self, *args, **kwargs)` | [`verl/utils/activation_offload.py`](../../verl/utils/activation_offload.py#L431) |
| `ActivationHandler._unpack_kwargs` | `def(self, flat_args, kwarg_keys)` | [`verl/utils/activation_offload.py`](../../verl/utils/activation_offload.py#L440) |
| `ActivationHandler.forward` | `def(self, module, forward_method, *args, **kwargs)` | [`verl/utils/activation_offload.py`](../../verl/utils/activation_offload.py#L463) |
| `ActivationHandler.post_forward` | `def(self, module)` | [`verl/utils/activation_offload.py`](../../verl/utils/activation_offload.py#L427) |
| `ActivationHandler.pre_forward` | `def(self, module)` | [`verl/utils/activation_offload.py`](../../verl/utils/activation_offload.py#L422) |
| `ActivationHandler.wrap_module_forward_method` | `def(self, module)` | [`verl/utils/activation_offload.py`](../../verl/utils/activation_offload.py#L479) |
| `ActivationHandler.wrap_module_forward_method.wrapped_method` | `def(model_self, *args, **kwargs)` | [`verl/utils/activation_offload.py`](../../verl/utils/activation_offload.py#L484) |
| `Actor.__init__` | `def(self)` | [`tests/ray_cpu/test_auto_padding.py`](../../tests/ray_cpu/test_auto_padding.py#L31) |
| `Actor.__init__` | `def(self)` | [`tests/ray_cpu/test_fused_workers.py`](../../tests/ray_cpu/test_fused_workers.py#L24) |
| `Actor.__init__` | `def(self)` | [`tests/ray_gpu/test_colocated_workers.py`](../../tests/ray_gpu/test_colocated_workers.py#L30) |
| `Actor.__init__` | `def(self)` | [`tests/ray_gpu/test_colocated_workers_fused.py`](../../tests/ray_gpu/test_colocated_workers_fused.py#L30) |
| `Actor.add` | `def(self, data)` | [`tests/ray_cpu/test_auto_padding.py`](../../tests/ray_cpu/test_auto_padding.py#L35) |
| `Actor.add` | `def(self, x)` | [`tests/ray_cpu/test_fused_workers.py`](../../tests/ray_cpu/test_fused_workers.py#L28) |
| `Actor.add` | `def(self, data)` | [`tests/ray_gpu/test_colocated_workers.py`](../../tests/ray_gpu/test_colocated_workers.py#L34) |
| `Actor.add` | `def(self, data)` | [`tests/ray_gpu/test_colocated_workers_fused.py`](../../tests/ray_gpu/test_colocated_workers_fused.py#L34) |
| `ActorRolloutRefWorker.__init__` | `def(self, config, role)` | [`verl/workers/fsdp_workers.py`](../../verl/workers/fsdp_workers.py#L106) |
| `ActorRolloutRefWorker.__init__` | `def(self, config, role)` | [`verl/workers/megatron_workers.py`](../../verl/workers/megatron_workers.py#L77) |
| `ActorRolloutRefWorker._build_model_optimizer` | `def(self, model_path, fsdp_config, optim_config, override_model_config, use_remove_padding, use_fused_kernels, enable_gradient_checkpointing, trust_remote_code, use_liger, role, enable_activation_offload)` | [`verl/workers/fsdp_workers.py`](../../verl/workers/fsdp_workers.py#L171) |
| `ActorRolloutRefWorker._build_model_optimizer` | `def(self, model_path, optim_config, override_model_config, override_transformer_config)` | [`verl/workers/megatron_workers.py`](../../verl/workers/megatron_workers.py#L142) |
| `ActorRolloutRefWorker._build_model_optimizer.megatron_actor_model_provider` | `def(pre_process, post_process)` | [`verl/workers/megatron_workers.py`](../../verl/workers/megatron_workers.py#L152) |
| `ActorRolloutRefWorker._build_rollout` | `def(self, trust_remote_code)` | [`verl/workers/fsdp_workers.py`](../../verl/workers/fsdp_workers.py#L396) |
| `ActorRolloutRefWorker._build_rollout` | `def(self, trust_remote_code)` | [`verl/workers/megatron_workers.py`](../../verl/workers/megatron_workers.py#L208) |
| `ActorRolloutRefWorker.begin_rollout_session` | `def(self)` | [`verl/workers/fsdp_workers.py`](../../verl/workers/fsdp_workers.py#L685) |
| `ActorRolloutRefWorker.compute_log_prob` | `def(self, data)` | [`verl/workers/fsdp_workers.py`](../../verl/workers/fsdp_workers.py#L720) |
| `ActorRolloutRefWorker.compute_log_prob` | `def(self, data)` | [`verl/workers/megatron_workers.py`](../../verl/workers/megatron_workers.py#L495) |
| `ActorRolloutRefWorker.compute_ref_log_prob` | `def(self, data)` | [`verl/workers/fsdp_workers.py`](../../verl/workers/fsdp_workers.py#L762) |
| `ActorRolloutRefWorker.compute_ref_log_prob` | `def(self, data)` | [`verl/workers/megatron_workers.py`](../../verl/workers/megatron_workers.py#L473) |
| `ActorRolloutRefWorker.compute_ref_topk_log_prob` | `def(self, data)` | [`verl/workers/fsdp_workers.py`](../../verl/workers/fsdp_workers.py#L797) |
| `ActorRolloutRefWorker.end_rollout_session` | `def(self)` | [`verl/workers/fsdp_workers.py`](../../verl/workers/fsdp_workers.py#L708) |
| `ActorRolloutRefWorker.generate_sequences` | `def(self, prompts)` | [`verl/workers/fsdp_workers.py`](../../verl/workers/fsdp_workers.py#L644) |
| `ActorRolloutRefWorker.generate_sequences` | `def(self, prompts)` | [`verl/workers/megatron_workers.py`](../../verl/workers/megatron_workers.py#L444) |
| `ActorRolloutRefWorker.init_model` | `def(self)` | [`verl/workers/fsdp_workers.py`](../../verl/workers/fsdp_workers.py#L506) |
| `ActorRolloutRefWorker.init_model` | `def(self)` | [`verl/workers/megatron_workers.py`](../../verl/workers/megatron_workers.py#L315) |
| `ActorRolloutRefWorker.load_checkpoint` | `def(self, local_path, hdfs_path, del_local_after_load)` | [`verl/workers/fsdp_workers.py`](../../verl/workers/fsdp_workers.py#L864) |
| `ActorRolloutRefWorker.load_checkpoint` | `def(self, checkpoint_path, hdfs_path, del_local_after_load)` | [`verl/workers/megatron_workers.py`](../../verl/workers/megatron_workers.py#L517) |
| `ActorRolloutRefWorker.load_pretrained_model` | `def(self, checkpoint_path, del_local_after_load)` | [`verl/workers/megatron_workers.py`](../../verl/workers/megatron_workers.py#L527) |
| `ActorRolloutRefWorker.save_checkpoint` | `def(self, local_path, hdfs_path, global_step, max_ckpt_to_keep)` | [`verl/workers/fsdp_workers.py`](../../verl/workers/fsdp_workers.py#L825) |
| `ActorRolloutRefWorker.save_checkpoint` | `def(self, checkpoint_path, hdfs_path, global_step, max_ckpt_to_keep)` | [`verl/workers/megatron_workers.py`](../../verl/workers/megatron_workers.py#L531) |
| `ActorRolloutRefWorker.update_actor` | `def(self, data)` | [`verl/workers/fsdp_workers.py`](../../verl/workers/fsdp_workers.py#L601) |
| `ActorRolloutRefWorker.update_actor` | `def(self, data)` | [`verl/workers/megatron_workers.py`](../../verl/workers/megatron_workers.py#L408) |
| `AdaptiveKLController.__init__` | `def(self, init_kl_coef, target_kl, horizon)` | [`recipe/spin/core_algos.py`](../../recipe/spin/core_algos.py#L28) |
| `AdaptiveKLController.__init__` | `def(self, init_kl_coef, target_kl, horizon)` | [`verl/trainer/ppo/core_algos.py`](../../verl/trainer/ppo/core_algos.py#L35) |
| `AdaptiveKLController.update` | `def(self, current_kl, n_steps)` | [`recipe/spin/core_algos.py`](../../recipe/spin/core_algos.py#L33) |
| `AdaptiveKLController.update` | `def(self, current_kl, n_steps)` | [`verl/trainer/ppo/core_algos.py`](../../verl/trainer/ppo/core_algos.py#L40) |
| `add_one` | `def(data)` | [`tests/ray_gpu/test_worker_group_basics.py`](../../tests/ray_gpu/test_worker_group_basics.py#L83) |
| `add_random_player_movement` | `def(room_state, room_structure, move_probability, continue_probability, max_steps)` | [`agent_system/environments/env_package/sokoban/sokoban/room_utils.py`](../../agent_system/environments/env_package/sokoban/sokoban/room_utils.py#L158) |
| `adjust_batch` | `def(config, data, mode)` | [`agent_system/multi_turn_rollout/utils.py`](../../agent_system/multi_turn_rollout/utils.py#L157) |
| `Agent.__init__` | `def(self, args)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/agent.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/agent.py#L32) |
| `Agent.__init__` | `def(self, model_name)` | [`examples/prompt_agent/gpt4o_alfworld.py`](../../examples/prompt_agent/gpt4o_alfworld.py#L28) |
| `Agent.act` | `def(self, states, valid_acts, method, state_strs, eps)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/agent.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/agent.py#L88) |
| `Agent.build_state` | `def(self, ob, info)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/agent.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/agent.py#L59) |
| `Agent.decode` | `def(self, act)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/agent.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/agent.py#L78) |
| `Agent.encode` | `def(self, observation, max_length)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/agent.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/agent.py#L71) |
| `Agent.encode_valids` | `def(self, valids, max_length)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/agent.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/agent.py#L83) |
| `Agent.get_action_from_gpt` | `def(self, obs)` | [`examples/prompt_agent/gpt4o_alfworld.py`](../../examples/prompt_agent/gpt4o_alfworld.py#L34) |
| `Agent.load` | `def(self)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/agent.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/agent.py#L152) |
| `Agent.save` | `def(self)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/agent.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/agent.py#L158) |
| `Agent.update` | `def(self, transitions, last_values, step, rewards_invdy)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/agent.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/agent.py#L120) |
| `agg` | `def(envs, attr)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/train_rl.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/train_rl.py#L66) |
| `agg_loss` | `def(loss_mat, loss_mask, loss_agg_mode)` | [`verl/trainer/ppo/core_algos.py`](../../verl/trainer/ppo/core_algos.py#L395) |
| `agg_loss_with_sample_weights` | `def(loss_mat, loss_mask, sample_weights, loss_agg_mode)` | [`verl/trainer/ppo/core_algos.py`](../../verl/trainer/ppo/core_algos.py#L431) |
| `alfworld_projection` | `def(actions, action_pools)` | [`agent_system/environments/env_package/alfworld/projection.py`](../../agent_system/environments/env_package/alfworld/projection.py#L19) |
| `AlfWorldEnvironmentManager.__init__` | `def(self, envs, projection_f, config)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L135) |
| `AlfWorldEnvironmentManager.__init__` | `def(self, envs, projection_f, config)` | [`recipe/hgpo/env_manager.py`](../../recipe/hgpo/env_manager.py#L120) |
| `AlfWorldEnvironmentManager._process_batch` | `def(self, batch_idx, total_batch_list, total_infos, success)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L215) |
| `AlfWorldEnvironmentManager._process_batch` | `def(self, batch_idx, total_batch_list, total_infos, success)` | [`recipe/hgpo/env_manager.py`](../../recipe/hgpo/env_manager.py#L202) |
| `AlfWorldEnvironmentManager._process_gamefile` | `def(self, gamefile, won_value, success)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L230) |
| `AlfWorldEnvironmentManager._process_gamefile` | `def(self, gamefile, won_value, success)` | [`recipe/hgpo/env_manager.py`](../../recipe/hgpo/env_manager.py#L217) |
| `AlfWorldEnvironmentManager.build_text_obs` | `def(self, text_obs, admissible_actions, init)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L181) |
| `AlfWorldEnvironmentManager.build_text_obs` | `def(self, text_obs, admissible_actions, init)` | [`recipe/hgpo/env_manager.py`](../../recipe/hgpo/env_manager.py#L166) |
| `AlfWorldEnvironmentManager.extract_task` | `def(self, text_obs)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L171) |
| `AlfWorldEnvironmentManager.extract_task` | `def(self, text_obs)` | [`recipe/hgpo/env_manager.py`](../../recipe/hgpo/env_manager.py#L156) |
| `AlfWorldEnvironmentManager.reset` | `def(self, kwargs)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L139) |
| `AlfWorldEnvironmentManager.reset` | `def(self, kwargs)` | [`recipe/hgpo/env_manager.py`](../../recipe/hgpo/env_manager.py#L124) |
| `AlfWorldEnvironmentManager.step` | `def(self, text_actions)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L151) |
| `AlfWorldEnvironmentManager.step` | `def(self, text_actions)` | [`recipe/hgpo/env_manager.py`](../../recipe/hgpo/env_manager.py#L136) |
| `AlfworldEnvs.__init__` | `def(self, alf_config_path, seed, env_num, group_n, resources_per_worker, is_train, env_kwargs)` | [`agent_system/environments/env_package/alfworld/envs.py`](../../agent_system/environments/env_package/alfworld/envs.py#L143) |
| `AlfworldEnvs.close` | `def(self)` | [`agent_system/environments/env_package/alfworld/envs.py`](../../agent_system/environments/env_package/alfworld/envs.py#L289) |
| `AlfworldEnvs.fast_forward` | `def(self, num_resets)` | [`agent_system/environments/env_package/alfworld/envs.py`](../../agent_system/environments/env_package/alfworld/envs.py#L237) |
| `AlfworldEnvs.fast_forward._complete` | `def(result)` | [`agent_system/environments/env_package/alfworld/envs.py`](../../agent_system/environments/env_package/alfworld/envs.py#L250) |
| `AlfworldEnvs.get_admissible_commands` | `def(self)` | [`agent_system/environments/env_package/alfworld/envs.py`](../../agent_system/environments/env_package/alfworld/envs.py#L282) |
| `AlfworldEnvs.getobs` | `def(self)` | [`agent_system/environments/env_package/alfworld/envs.py`](../../agent_system/environments/env_package/alfworld/envs.py#L268) |
| `AlfworldEnvs.probe_game_iterator` | `def(self)` | [`agent_system/environments/env_package/alfworld/envs.py`](../../agent_system/environments/env_package/alfworld/envs.py#L233) |
| `AlfworldEnvs.reset` | `def(self)` | [`agent_system/environments/env_package/alfworld/envs.py`](../../agent_system/environments/env_package/alfworld/envs.py#L203) |
| `AlfworldEnvs.step` | `def(self, actions)` | [`agent_system/environments/env_package/alfworld/envs.py`](../../agent_system/environments/env_package/alfworld/envs.py#L167) |
| `AlfworldWorker.__init__` | `def(self, config, seed, base_env)` | [`agent_system/environments/env_package/alfworld/envs.py`](../../agent_system/environments/env_package/alfworld/envs.py#L61) |
| `AlfworldWorker._find_game_iterator` | `def(self)` | [`agent_system/environments/env_package/alfworld/envs.py`](../../agent_system/environments/env_package/alfworld/envs.py#L88) |
| `AlfworldWorker.getobs` | `def(self)` | [`agent_system/environments/env_package/alfworld/envs.py`](../../agent_system/environments/env_package/alfworld/envs.py#L79) |
| `AlfworldWorker.reset` | `def(self)` | [`agent_system/environments/env_package/alfworld/envs.py`](../../agent_system/environments/env_package/alfworld/envs.py#L73) |
| `AlfworldWorker.skip_games` | `def(self, num_resets)` | [`agent_system/environments/env_package/alfworld/envs.py`](../../agent_system/environments/env_package/alfworld/envs.py#L112) |
| `AlfworldWorker.step` | `def(self, action)` | [`agent_system/environments/env_package/alfworld/envs.py`](../../agent_system/environments/env_package/alfworld/envs.py#L65) |
| `all_gather_data_proto` | `def(data, process_group)` | [`verl/protocol.py`](../../verl/protocol.py#L887) |
| `all_gather_tensor` | `def(local_tensor, group, async_op)` | [`verl/utils/ulysses.py`](../../verl/utils/ulysses.py#L155) |
| `all_to_all_tensor` | `def(local_input, scatter_dim, gather_dim, group, async_op)` | [`verl/utils/ulysses.py`](../../verl/utils/ulysses.py#L133) |
| `all_to_all_tensor.wait` | `def()` | [`verl/utils/ulysses.py`](../../verl/utils/ulysses.py#L147) |
| `allgather_dict_tensors` | `def(tensors, size, group, dim)` | [`verl/utils/torch_functional.py`](../../verl/utils/torch_functional.py#L230) |
| `AllGatherPPModel.__call__` | `def(self, *inputs, **kwargs)` | [`verl/workers/sharding_manager/megatron_vllm.py`](../../verl/workers/sharding_manager/megatron_vllm.py#L183) |
| `AllGatherPPModel.__init__` | `def(self, model_provider, use_distributed_optimizer)` | [`verl/workers/sharding_manager/megatron_vllm.py`](../../verl/workers/sharding_manager/megatron_vllm.py#L57) |
| `AllGatherPPModel._build_param_buffer` | `def(self, pp_rank)` | [`verl/workers/sharding_manager/megatron_vllm.py`](../../verl/workers/sharding_manager/megatron_vllm.py#L106) |
| `AllGatherPPModel._build_param_references` | `def(self, pp_rank, maintain_weight)` | [`verl/workers/sharding_manager/megatron_vllm.py`](../../verl/workers/sharding_manager/megatron_vllm.py#L122) |
| `AllGatherPPModel._load_params_to_cuda` | `def(self, pp_rank, to_empty)` | [`verl/workers/sharding_manager/megatron_vllm.py`](../../verl/workers/sharding_manager/megatron_vllm.py#L128) |
| `AllGatherPPModel._offload_params_to_cpu` | `def(self, pp_rank, to_empty)` | [`verl/workers/sharding_manager/megatron_vllm.py`](../../verl/workers/sharding_manager/megatron_vllm.py#L138) |
| `AllGatherPPModel.allgather_params` | `def(self)` | [`verl/workers/sharding_manager/megatron_vllm.py`](../../verl/workers/sharding_manager/megatron_vllm.py#L154) |
| `AllGatherPPModel.eval` | `def(self)` | [`verl/workers/sharding_manager/megatron_vllm.py`](../../verl/workers/sharding_manager/megatron_vllm.py#L186) |
| `AllGatherPPModel.forward` | `def(self, *inputs, **kwargs)` | [`verl/workers/sharding_manager/megatron_vllm.py`](../../verl/workers/sharding_manager/megatron_vllm.py#L164) |
| `AllGatherPPModel.get_all_params` | `def(self)` | [`verl/workers/sharding_manager/megatron_vllm.py`](../../verl/workers/sharding_manager/megatron_vllm.py#L200) |
| `AllGatherPPModel.load_params_to_cuda` | `def(self, to_empty)` | [`verl/workers/sharding_manager/megatron_vllm.py`](../../verl/workers/sharding_manager/megatron_vllm.py#L148) |
| `AllGatherPPModel.offload_params_to_cpu` | `def(self, to_empty)` | [`verl/workers/sharding_manager/megatron_vllm.py`](../../verl/workers/sharding_manager/megatron_vllm.py#L194) |
| `AllGatherPPModel.pp_group` | `def(self)` | [`verl/workers/sharding_manager/megatron_vllm.py`](../../verl/workers/sharding_manager/megatron_vllm.py#L240) |
| `AllGatherPPModel.pp_models` | `def(self)` | [`verl/workers/sharding_manager/megatron_vllm.py`](../../verl/workers/sharding_manager/megatron_vllm.py#L244) |
| `AllGatherPPModel.pp_rank` | `def(self)` | [`verl/workers/sharding_manager/megatron_vllm.py`](../../verl/workers/sharding_manager/megatron_vllm.py#L236) |
| `AllGatherPPModel.pp_size` | `def(self)` | [`verl/workers/sharding_manager/megatron_vllm.py`](../../verl/workers/sharding_manager/megatron_vllm.py#L232) |
| `AllGatherPPModel.this_rank_models` | `def(self)` | [`verl/workers/sharding_manager/megatron_vllm.py`](../../verl/workers/sharding_manager/megatron_vllm.py#L228) |
| `AllGatherPPModel.train` | `def(self)` | [`verl/workers/sharding_manager/megatron_vllm.py`](../../verl/workers/sharding_manager/megatron_vllm.py#L190) |
| `AllGatherPPModel.update_this_rank_models` | `def(self, new_models)` | [`verl/workers/sharding_manager/megatron_vllm.py`](../../verl/workers/sharding_manager/megatron_vllm.py#L223) |
| `annotate` | `def(attr_path)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/attributes/annotate.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/attributes/annotate.py#L16) |
| `append_to_dict` | `def(data, new_data)` | [`verl/utils/py_functional.py`](../../verl/utils/py_functional.py#L159) |
| `apply_fsdp2` | `def(model, fsdp_kwargs, config)` | [`verl/utils/fsdp_utils.py`](../../verl/utils/fsdp_utils.py#L425) |
| `apply_invalid_action_penalty` | `def(data, invalid_action_penalty_coef)` | [`recipe/hgpo/hgpo_ray_trainer.py`](../../recipe/hgpo/hgpo_ray_trainer.py#L203) |
| `apply_invalid_action_penalty` | `def(data, invalid_action_penalty_coef, invalid_action_penalty_coef_by_task)` | [`verl/trainer/ppo/ray_trainer.py`](../../verl/trainer/ppo/ray_trainer.py#L227) |
| `apply_kl_penalty` | `def(data, kl_ctrl, kl_penalty, multi_turn)` | [`recipe/hgpo/hgpo_ray_trainer.py`](../../recipe/hgpo/hgpo_ray_trainer.py#L155) |
| `apply_kl_penalty` | `def(data, kl_ctrl, kl_penalty)` | [`recipe/spin/spin_trainer.py`](../../recipe/spin/spin_trainer.py#L265) |
| `apply_kl_penalty` | `def(data, kl_ctrl, kl_penalty, multi_turn)` | [`verl/trainer/ppo/ray_trainer.py`](../../verl/trainer/ppo/ray_trainer.py#L156) |
| `apply_monkey_patch` | `def(model, ulysses_sp_size, use_remove_padding, use_fused_kernels, fused_kernels_backend)` | [`verl/models/transformers/monkey_patch.py`](../../verl/models/transformers/monkey_patch.py#L248) |
| `apply_monkey_patch.state_dict` | `def(self, *args, **kwargs)` | [`verl/models/transformers/monkey_patch.py`](../../verl/models/transformers/monkey_patch.py#L285) |
| `apply_patch` | `def()` | [`verl/models/mcore/patch_v012.py`](../../verl/models/mcore/patch_v012.py#L20) |
| `apply_patch.patch_get_query_key_value_tensors` | `def(self, hidden_states, key_value_states, position_ids, packed_seq_params, inference_context)` | [`verl/models/mcore/patch_v012.py`](../../verl/models/mcore/patch_v012.py#L25) |
| `apply_patch.patch_get_query_key_value_tensors.qkv_up_proj_and_rope_apply` | `def(q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb)` | [`verl/models/mcore/patch_v012.py`](../../verl/models/mcore/patch_v012.py#L106) |
| `apply_rotary_pos_emb` | `def(q, k, cos, sin, position_ids)` | [`verl/models/llama/megatron/layers/parallel_attention.py`](../../verl/models/llama/megatron/layers/parallel_attention.py#L149) |
| `apply_rotary_pos_emb` | `def(q, k, cos, sin, position_ids)` | [`verl/models/qwen2/megatron/layers/parallel_attention.py`](../../verl/models/qwen2/megatron/layers/parallel_attention.py#L126) |
| `apply_rotary_pos_emb` | `def(q, k, cos, sin, position_ids, unsqueeze_dim)` | [`verl/models/transformers/kimi_vl.py`](../../verl/models/transformers/kimi_vl.py#L78) |
| `apply_rotary_pos_emb_rmpad` | `def(q, k, cos, sin, position_ids, indices, sequence_length)` | [`verl/models/llama/megatron/layers/parallel_attention.py`](../../verl/models/llama/megatron/layers/parallel_attention.py#L326) |
| `apply_rotary_pos_emb_rmpad` | `def(q, k, cos, sin, position_ids, indices, sequence_length)` | [`verl/models/qwen2/megatron/layers/parallel_attention.py`](../../verl/models/qwen2/megatron/layers/parallel_attention.py#L272) |
| `apply_rotary_pos_emb_rmpad_flash` | `def(q, k, cos, sin, cu_seqlens, max_seqlen)` | [`verl/models/llama/megatron/layers/parallel_attention.py`](../../verl/models/llama/megatron/layers/parallel_attention.py#L344) |
| `apply_rotary_pos_emb_rmpad_flash` | `def(q, k, cos, sin, cu_seqlens, max_seqlen)` | [`verl/models/qwen2/megatron/layers/parallel_attention.py`](../../verl/models/qwen2/megatron/layers/parallel_attention.py#L290) |
| `appworld_projection` | `def(actions)` | [`agent_system/environments/env_package/appworld/projection.py`](../../agent_system/environments/env_package/appworld/projection.py#L22) |
| `AppWorldEnvironmentManager.__init__` | `def(self, envs, projection_f, config)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L521) |
| `AppWorldEnvironmentManager.__init__` | `def(self, envs, projection_f, config)` | [`recipe/hgpo/env_manager.py`](../../recipe/hgpo/env_manager.py#L509) |
| `AppWorldEnvironmentManager.build_text_obs` | `def(self, text_obs, init)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L557) |
| `AppWorldEnvironmentManager.build_text_obs` | `def(self, text_obs, init)` | [`recipe/hgpo/env_manager.py`](../../recipe/hgpo/env_manager.py#L545) |
| `AppWorldEnvironmentManager.reset` | `def(self, kwargs)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L525) |
| `AppWorldEnvironmentManager.reset` | `def(self, kwargs)` | [`recipe/hgpo/env_manager.py`](../../recipe/hgpo/env_manager.py#L513) |
| `AppWorldEnvironmentManager.step` | `def(self, text_actions)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L536) |
| `AppWorldEnvironmentManager.step` | `def(self, text_actions)` | [`recipe/hgpo/env_manager.py`](../../recipe/hgpo/env_manager.py#L524) |
| `AppWorldEnvs.__init__` | `def(self, dataset_name, max_interactions, seed, env_num, group_n, start_server_id, resources_per_worker, port_file)` | [`agent_system/environments/env_package/appworld/envs.py`](../../agent_system/environments/env_package/appworld/envs.py#L110) |
| `AppWorldEnvs.close` | `def(self)` | [`agent_system/environments/env_package/appworld/envs.py`](../../agent_system/environments/env_package/appworld/envs.py#L219) |
| `AppWorldEnvs.render` | `def(self)` | [`agent_system/environments/env_package/appworld/envs.py`](../../agent_system/environments/env_package/appworld/envs.py#L234) |
| `AppWorldEnvs.reset` | `def(self)` | [`agent_system/environments/env_package/appworld/envs.py`](../../agent_system/environments/env_package/appworld/envs.py#L191) |
| `AppWorldEnvs.step` | `def(self, actions)` | [`agent_system/environments/env_package/appworld/envs.py`](../../agent_system/environments/env_package/appworld/envs.py#L159) |
| `AppWorldWorker.__init__` | `def(self, worker_id, max_interactions, port)` | [`agent_system/environments/env_package/appworld/envs.py`](../../agent_system/environments/env_package/appworld/envs.py#L47) |
| `AppWorldWorker.close` | `def(self)` | [`agent_system/environments/env_package/appworld/envs.py`](../../agent_system/environments/env_package/appworld/envs.py#L98) |
| `AppWorldWorker.reset` | `def(self, task_id)` | [`agent_system/environments/env_package/appworld/envs.py`](../../agent_system/environments/env_package/appworld/envs.py#L55) |
| `AppWorldWorker.step` | `def(self, action)` | [`agent_system/environments/env_package/appworld/envs.py`](../../agent_system/environments/env_package/appworld/envs.py#L76) |
| `are_equal_under_sympy` | `def(ground_truth_normalized, given_normalized)` | [`verl/utils/reward_score/prime_math/__init__.py`](../../verl/utils/reward_score/prime_math/__init__.py#L213) |
| `are_lists_similar` | `def(a, b)` | [`tests/workers/rollout/test_vllm_hf_loader.py`](../../tests/workers/rollout/test_vllm_hf_loader.py#L47) |
| `are_lists_similar` | `def(a, b)` | [`tests/workers/rollout/test_vllm_spmd.py`](../../tests/workers/rollout/test_vllm_spmd.py#L49) |
| `are_lists_similar` | `def(a, b, threshold)` | [`tests/workers/rollout/utils_sglang.py`](../../tests/workers/rollout/utils_sglang.py#L40) |
| `are_similar` | `def(a, b, threshold)` | [`gigpo/core_gigpo.py`](../../gigpo/core_gigpo.py#L72) |
| `assign_uids` | `def(n_traj, rollout_n, uid_factory)` | [`agent_system/multi_turn_rollout/async_rollout_core.py`](../../agent_system/multi_turn_rollout/async_rollout_core.py#L92) |
| `async_server_class` | `def(rollout_backend)` | [`verl/workers/rollout/async_server.py`](../../verl/workers/rollout/async_server.py#L338) |
| `async_to_comparable` | `def(result)` | [`tests/ray_cpu/test_async_rollout_equivalence.py`](../../tests/ray_cpu/test_async_rollout_equivalence.py#L112) |
| `AsyncActorRolloutRefWorker._build_rollout` | `def(self, trust_remote_code)` | [`verl/workers/fsdp_workers.py`](../../verl/workers/fsdp_workers.py#L1514) |
| `AsyncActorRolloutRefWorker.execute_method` | `def(self, method, *args, **kwargs)` | [`verl/workers/fsdp_workers.py`](../../verl/workers/fsdp_workers.py#L1534) |
| `AsyncActorRolloutRefWorker.generate_sequences` | `def(self, prompts)` | [`verl/workers/fsdp_workers.py`](../../verl/workers/fsdp_workers.py#L1530) |
| `AsyncActorRolloutRefWorker.offload` | `def(self)` | [`verl/workers/fsdp_workers.py`](../../verl/workers/fsdp_workers.py#L1545) |
| `AsyncActorRolloutRefWorker.resume` | `def(self)` | [`verl/workers/fsdp_workers.py`](../../verl/workers/fsdp_workers.py#L1541) |
| `AsyncDoubleBufferGroupOffloadHandler.__init__` | `def(self, num_offload_group, num_model_group, tensor_need_offloading_checker)` | [`verl/utils/activation_offload.py`](../../verl/utils/activation_offload.py#L221) |
| `AsyncDoubleBufferGroupOffloadHandler.bulk_offload_group` | `def(self, group_to_offload)` | [`verl/utils/activation_offload.py`](../../verl/utils/activation_offload.py#L294) |
| `AsyncDoubleBufferGroupOffloadHandler.bulk_reload_group` | `def(self, group_to_reload)` | [`verl/utils/activation_offload.py`](../../verl/utils/activation_offload.py#L351) |
| `AsyncDoubleBufferGroupOffloadHandler.on_group_commit_backward` | `def(self)` | [`verl/utils/activation_offload.py`](../../verl/utils/activation_offload.py#L369) |
| `AsyncDoubleBufferGroupOffloadHandler.on_group_commit_forward` | `def(self)` | [`verl/utils/activation_offload.py`](../../verl/utils/activation_offload.py#L343) |
| `AsyncDoubleBufferGroupOffloadHandler.synchronize_on_group_commit_forward` | `def(self, current_group)` | [`verl/utils/activation_offload.py`](../../verl/utils/activation_offload.py#L315) |
| `AsyncDoubleBufferGroupOffloadHandler.tensor_pop` | `def(self, tensor_tag, **kwargs)` | [`verl/utils/activation_offload.py`](../../verl/utils/activation_offload.py#L281) |
| `AsyncDoubleBufferGroupOffloadHandler.tensor_push` | `def(self, tensor, **kwargs)` | [`verl/utils/activation_offload.py`](../../verl/utils/activation_offload.py#L256) |
| `AsyncLLMServerManager.__init__` | `def(self, config, worker_group)` | [`verl/workers/rollout/async_server.py`](../../verl/workers/rollout/async_server.py#L221) |
| `AsyncLLMServerManager._init_chat_scheduler` | `def(self)` | [`verl/workers/rollout/async_server.py`](../../verl/workers/rollout/async_server.py#L283) |
| `AsyncLLMServerManager.generate_sequences` | `def(self, prompts, **sampling_params)` | [`verl/workers/rollout/async_server.py`](../../verl/workers/rollout/async_server.py#L330) |
| `AsyncLLMServerManager.sleep` | `def(self)` | [`verl/workers/rollout/async_server.py`](../../verl/workers/rollout/async_server.py#L304) |
| `AsyncLLMServerManager.submit_chat_completions` | `def(self, callback, callback_additional_info, **chat_complete_request)` | [`verl/workers/rollout/async_server.py`](../../verl/workers/rollout/async_server.py#L308) |
| `AsyncLLMServerManager.wake_up` | `def(self)` | [`verl/workers/rollout/async_server.py`](../../verl/workers/rollout/async_server.py#L300) |
| `AsyncRolloutRequest.add_assistant_message` | `def(self, tokenizer, content, tool_calls, format, already_over_long)` | [`verl/workers/rollout/schemas.py`](../../verl/workers/rollout/schemas.py#L115) |
| `AsyncRolloutRequest.add_tool_response_message` | `def(self, tokenizer, content, last_tool, format)` | [`verl/workers/rollout/schemas.py`](../../verl/workers/rollout/schemas.py#L170) |
| `AsyncRolloutRequest.finalize` | `def(self, tokenizer, reward_scores, finish_reason_type)` | [`verl/workers/rollout/schemas.py`](../../verl/workers/rollout/schemas.py#L216) |
| `AsyncRolloutRequest.get_generation_prompt` | `def(self, tokenizer)` | [`verl/workers/rollout/schemas.py`](../../verl/workers/rollout/schemas.py#L107) |
| `AsyncRolloutRequest.truncate_output_ids` | `def(self, tokenizer)` | [`verl/workers/rollout/schemas.py`](../../verl/workers/rollout/schemas.py#L235) |
| `AsyncRolloutRequest.update_metrics` | `def(self, metrics, tool_id)` | [`verl/workers/rollout/schemas.py`](../../verl/workers/rollout/schemas.py#L208) |
| `AsyncServerBase.__init__` | `def(self)` | [`verl/workers/rollout/async_server.py`](../../verl/workers/rollout/async_server.py#L53) |
| `AsyncServerBase._start_fastapi_server` | `async def(self)` | [`verl/workers/rollout/async_server.py`](../../verl/workers/rollout/async_server.py#L59) |
| `AsyncServerBase._start_fastapi_server.lifespan` | `async def(app)` | [`verl/workers/rollout/async_server.py`](../../verl/workers/rollout/async_server.py#L61) |
| `AsyncServerBase.chat_completion` | `async def(self, raw_request)` | [`verl/workers/rollout/async_server.py`](../../verl/workers/rollout/async_server.py#L85) |
| `AsyncServerBase.get_server_address` | `async def(self)` | [`verl/workers/rollout/async_server.py`](../../verl/workers/rollout/async_server.py#L79) |
| `AsyncServerBase.init_engine` | `async def(self)` | [`verl/workers/rollout/async_server.py`](../../verl/workers/rollout/async_server.py#L93) |
| `AsyncServerBase.sleep` | `async def(self)` | [`verl/workers/rollout/async_server.py`](../../verl/workers/rollout/async_server.py#L103) |
| `AsyncServerBase.wake_up` | `async def(self)` | [`verl/workers/rollout/async_server.py`](../../verl/workers/rollout/async_server.py#L98) |
| `AsyncSglangServer.__init__` | `def(self, config, dp_size, dp_rank, wg_prefix)` | [`verl/workers/rollout/sglang_rollout/async_sglang_server.py`](../../verl/workers/rollout/sglang_rollout/async_sglang_server.py#L30) |
| `AsyncSglangServer.chat_completion` | `async def(self, raw_request)` | [`verl/workers/rollout/sglang_rollout/async_sglang_server.py`](../../verl/workers/rollout/sglang_rollout/async_sglang_server.py#L52) |
| `AsyncSglangServer.init_engine` | `async def(self)` | [`verl/workers/rollout/sglang_rollout/async_sglang_server.py`](../../verl/workers/rollout/sglang_rollout/async_sglang_server.py#L40) |
| `AsyncSglangServer.sleep` | `async def(self)` | [`verl/workers/rollout/sglang_rollout/async_sglang_server.py`](../../verl/workers/rollout/sglang_rollout/async_sglang_server.py#L70) |
| `AsyncSglangServer.wake_up` | `async def(self)` | [`verl/workers/rollout/sglang_rollout/async_sglang_server.py`](../../verl/workers/rollout/sglang_rollout/async_sglang_server.py#L66) |
| `AsyncvLLMServer.__init__` | `def(self, config, vllm_dp_size, vllm_dp_rank, wg_prefix)` | [`verl/workers/rollout/vllm_rollout/vllm_async_server.py`](../../verl/workers/rollout/vllm_rollout/vllm_async_server.py#L122) |
| `AsyncvLLMServer.chat_completion` | `async def(self, raw_request)` | [`verl/workers/rollout/vllm_rollout/vllm_async_server.py`](../../verl/workers/rollout/vllm_rollout/vllm_async_server.py#L206) |
| `AsyncvLLMServer.chat_completion_generator` | `async def(self, request)` | [`verl/workers/rollout/vllm_rollout/vllm_async_server.py`](../../verl/workers/rollout/vllm_rollout/vllm_async_server.py#L223) |
| `AsyncvLLMServer.init_engine` | `async def(self)` | [`verl/workers/rollout/vllm_rollout/vllm_async_server.py`](../../verl/workers/rollout/vllm_rollout/vllm_async_server.py#L138) |
| `AsyncvLLMServer.sleep` | `async def(self)` | [`verl/workers/rollout/vllm_rollout/vllm_async_server.py`](../../verl/workers/rollout/vllm_rollout/vllm_async_server.py#L248) |
| `AsyncvLLMServer.wake_up` | `async def(self)` | [`verl/workers/rollout/vllm_rollout/vllm_async_server.py`](../../verl/workers/rollout/vllm_rollout/vllm_async_server.py#L245) |
| `bart_predict` | `def(input, model, skip_special_tokens, **kwargs)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/test.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/test.py#L21) |
| `bart_predict` | `def(input)` | [`agent_system/environments/env_package/webshop/webshop/transfer/app.py`](../../agent_system/environments/env_package/webshop/webshop/transfer/app.py#L66) |
| `BaseCheckpointManager.__init__` | `def(self, model, optimizer, lr_scheduler, processing_class, checkpoint_contents)` | [`verl/utils/checkpoint/checkpoint_manager.py`](../../verl/utils/checkpoint/checkpoint_manager.py#L44) |
| `BaseCheckpointManager.checkpath` | `def(local_path, hdfs_path)` | [`verl/utils/checkpoint/checkpoint_manager.py`](../../verl/utils/checkpoint/checkpoint_manager.py#L73) |
| `BaseCheckpointManager.get_rng_state` | `def()` | [`verl/utils/checkpoint/checkpoint_manager.py`](../../verl/utils/checkpoint/checkpoint_manager.py#L109) |
| `BaseCheckpointManager.load_checkpoint` | `def(self, local_path, hdfs_path, del_local_after_load)` | [`verl/utils/checkpoint/checkpoint_manager.py`](../../verl/utils/checkpoint/checkpoint_manager.py#L66) |
| `BaseCheckpointManager.load_rng_state` | `def(rng_state)` | [`verl/utils/checkpoint/checkpoint_manager.py`](../../verl/utils/checkpoint/checkpoint_manager.py#L124) |
| `BaseCheckpointManager.local_mkdir` | `def(path)` | [`verl/utils/checkpoint/checkpoint_manager.py`](../../verl/utils/checkpoint/checkpoint_manager.py#L88) |
| `BaseCheckpointManager.remove_previous_save_local_path` | `def(self, path)` | [`verl/utils/checkpoint/checkpoint_manager.py`](../../verl/utils/checkpoint/checkpoint_manager.py#L77) |
| `BaseCheckpointManager.save_checkpoint` | `def(self, local_path, hdfs_path, global_step, max_ckpt_to_keep)` | [`verl/utils/checkpoint/checkpoint_manager.py`](../../verl/utils/checkpoint/checkpoint_manager.py#L69) |
| `BaseDiscreteActionEnv.copy` | `def(self)` | [`agent_system/environments/env_package/sokoban/sokoban/base.py`](../../agent_system/environments/env_package/sokoban/sokoban/base.py#L307) |
| `BaseDiscreteActionEnv.finished` | `def(self)` | [`agent_system/environments/env_package/sokoban/sokoban/base.py`](../../agent_system/environments/env_package/sokoban/sokoban/base.py#L289) |
| `BaseDiscreteActionEnv.get_all_actions` | `def(self)` | [`agent_system/environments/env_package/sokoban/sokoban/base.py`](../../agent_system/environments/env_package/sokoban/sokoban/base.py#L251) |
| `BaseDiscreteActionEnv.parse_update_info_to_obs` | `def(update_info, action_is_valid)` | [`agent_system/environments/env_package/sokoban/sokoban/base.py`](../../agent_system/environments/env_package/sokoban/sokoban/base.py#L234) |
| `BaseDiscreteActionEnv.render` | `def(self, mode)` | [`agent_system/environments/env_package/sokoban/sokoban/base.py`](../../agent_system/environments/env_package/sokoban/sokoban/base.py#L294) |
| `BaseDiscreteActionEnv.reset` | `def(self, mode, seed)` | [`agent_system/environments/env_package/sokoban/sokoban/base.py`](../../agent_system/environments/env_package/sokoban/sokoban/base.py#L257) |
| `BaseDiscreteActionEnv.step` | `def(self, action)` | [`agent_system/environments/env_package/sokoban/sokoban/base.py`](../../agent_system/environments/env_package/sokoban/sokoban/base.py#L271) |
| `BaseDiscreteActionEnv.success` | `def(self)` | [`agent_system/environments/env_package/sokoban/sokoban/base.py`](../../agent_system/environments/env_package/sokoban/sokoban/base.py#L284) |
| `BaseEnv.__init__` | `def(self)` | [`agent_system/environments/env_package/sokoban/sokoban/base.py`](../../agent_system/environments/env_package/sokoban/sokoban/base.py#L17) |
| `BaseEnv._copy_tracking_variables` | `def(self, other)` | [`agent_system/environments/env_package/sokoban/sokoban/base.py`](../../agent_system/environments/env_package/sokoban/sokoban/base.py#L71) |
| `BaseEnv._extract_answer` | `def(text)` | [`agent_system/environments/env_package/sokoban/sokoban/base.py`](../../agent_system/environments/env_package/sokoban/sokoban/base.py#L25) |
| `BaseEnv._reset_tracking_variables` | `def(self)` | [`agent_system/environments/env_package/sokoban/sokoban/base.py`](../../agent_system/environments/env_package/sokoban/sokoban/base.py#L33) |
| `BaseEnv._update_tracking_variables` | `def(self, response, action, action_is_valid, action_is_effective, reward)` | [`agent_system/environments/env_package/sokoban/sokoban/base.py`](../../agent_system/environments/env_package/sokoban/sokoban/base.py#L48) |
| `BaseEnv.copy` | `def(self)` | [`agent_system/environments/env_package/sokoban/sokoban/base.py`](../../agent_system/environments/env_package/sokoban/sokoban/base.py#L214) |
| `BaseEnv.execute_predictions` | `def(cls, envs, predictions, prediction_ids, tokenizer)` | [`agent_system/environments/env_package/sokoban/sokoban/base.py`](../../agent_system/environments/env_package/sokoban/sokoban/base.py#L94) |
| `BaseEnv.finished` | `def(self)` | [`agent_system/environments/env_package/sokoban/sokoban/base.py`](../../agent_system/environments/env_package/sokoban/sokoban/base.py#L204) |
| `BaseEnv.formulate_output` | `def(env_feedback, done)` | [`agent_system/environments/env_package/sokoban/sokoban/base.py`](../../agent_system/environments/env_package/sokoban/sokoban/base.py#L80) |
| `BaseEnv.get_tracking_variables` | `def(self)` | [`agent_system/environments/env_package/sokoban/sokoban/base.py`](../../agent_system/environments/env_package/sokoban/sokoban/base.py#L39) |
| `BaseEnv.parse_update_info_to_obs` | `def(update_info, action_is_valid)` | [`agent_system/environments/env_package/sokoban/sokoban/base.py`](../../agent_system/environments/env_package/sokoban/sokoban/base.py#L159) |
| `BaseEnv.render` | `def(self, mode)` | [`agent_system/environments/env_package/sokoban/sokoban/base.py`](../../agent_system/environments/env_package/sokoban/sokoban/base.py#L209) |
| `BaseEnv.reset` | `def(self, seed)` | [`agent_system/environments/env_package/sokoban/sokoban/base.py`](../../agent_system/environments/env_package/sokoban/sokoban/base.py#L173) |
| `BaseEnv.step` | `def(self, action)` | [`agent_system/environments/env_package/sokoban/sokoban/base.py`](../../agent_system/environments/env_package/sokoban/sokoban/base.py#L186) |
| `BaseEnv.success` | `def(self)` | [`agent_system/environments/env_package/sokoban/sokoban/base.py`](../../agent_system/environments/env_package/sokoban/sokoban/base.py#L199) |
| `BaseMemory.__getitem__` | `def(self, idx)` | [`agent_system/memory/base.py`](../../agent_system/memory/base.py#L31) |
| `BaseMemory.__len__` | `def(self)` | [`agent_system/memory/base.py`](../../agent_system/memory/base.py#L26) |
| `BaseMemory.fetch` | `def(self, step)` | [`agent_system/memory/base.py`](../../agent_system/memory/base.py#L50) |
| `BaseMemory.reset` | `def(self, batch_size)` | [`agent_system/memory/base.py`](../../agent_system/memory/base.py#L36) |
| `BaseMemory.store` | `def(self, record)` | [`agent_system/memory/base.py`](../../agent_system/memory/base.py#L43) |
| `BaseModelInitializer.__init__` | `def(self, tfconfig, hf_config)` | [`verl/models/mcore/model_initializer.py`](../../verl/models/mcore/model_initializer.py#L30) |
| `BaseModelInitializer.get_rope_scaling_args` | `def(self)` | [`verl/models/mcore/model_initializer.py`](../../verl/models/mcore/model_initializer.py#L41) |
| `BaseModelInitializer.get_transformer_layer_spec` | `def(self, vp_stage)` | [`verl/models/mcore/model_initializer.py`](../../verl/models/mcore/model_initializer.py#L36) |
| `BaseModelInitializer.initialize` | `def(self, pre_process, post_process, share_embeddings_and_output_weights, value, **extra_kwargs)` | [`verl/models/mcore/model_initializer.py`](../../verl/models/mcore/model_initializer.py#L50) |
| `BaseModelMerger.__init__` | `def(self, config)` | [`scripts/model_merger.py`](../../scripts/model_merger.py#L97) |
| `BaseModelMerger.get_transformers_auto_model_class` | `def(self)` | [`scripts/model_merger.py`](../../scripts/model_merger.py#L107) |
| `BaseModelMerger.merge_and_save` | `def(self)` | [`scripts/model_merger.py`](../../scripts/model_merger.py#L160) |
| `BaseModelMerger.patch_model_generation_config` | `def(self, model)` | [`scripts/model_merger.py`](../../scripts/model_merger.py#L117) |
| `BaseModelMerger.save_hf_model_and_tokenizer` | `def(self, state_dict)` | [`scripts/model_merger.py`](../../scripts/model_merger.py#L131) |
| `BaseModelMerger.upload_to_huggingface` | `def(self)` | [`scripts/model_merger.py`](../../scripts/model_merger.py#L152) |
| `BasePolicy.__init__` | `def(self)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/models/models.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/models/models.py#L11) |
| `BasePolicy.forward` | `def(observation, available_actions)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/models/models.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/models/models.py#L14) |
| `BasePPOActor.__init__` | `def(self, config)` | [`verl/workers/actor/base.py`](../../verl/workers/actor/base.py#L29) |
| `BasePPOActor.compute_log_prob` | `def(self, data)` | [`verl/workers/actor/base.py`](../../verl/workers/actor/base.py#L40) |
| `BasePPOActor.update_policy` | `def(self, data)` | [`verl/workers/actor/base.py`](../../verl/workers/actor/base.py#L55) |
| `BasePPOCritic.__init__` | `def(self, config)` | [`verl/workers/critic/base.py`](../../verl/workers/critic/base.py#L28) |
| `BasePPOCritic.compute_values` | `def(self, data)` | [`verl/workers/critic/base.py`](../../verl/workers/critic/base.py#L33) |
| `BasePPOCritic.update_critic` | `def(self, data)` | [`verl/workers/critic/base.py`](../../verl/workers/critic/base.py#L38) |
| `BasePPORewardModel.__init__` | `def(self, config)` | [`verl/workers/reward_model/base.py`](../../verl/workers/reward_model/base.py#L24) |
| `BasePPORewardModel.compute_reward` | `def(self, data)` | [`verl/workers/reward_model/base.py`](../../verl/workers/reward_model/base.py#L28) |
| `BaseRetriever.__init__` | `def(self, config)` | [`examples/search/retriever/retrieval_server.py`](../../examples/search/retriever/retrieval_server.py#L114) |
| `BaseRetriever._batch_search` | `def(self, query_list, num, return_score)` | [`examples/search/retriever/retrieval_server.py`](../../examples/search/retriever/retrieval_server.py#L125) |
| `BaseRetriever._search` | `def(self, query, num, return_score)` | [`examples/search/retriever/retrieval_server.py`](../../examples/search/retriever/retrieval_server.py#L122) |
| `BaseRetriever.batch_search` | `def(self, query_list, num, return_score)` | [`examples/search/retriever/retrieval_server.py`](../../examples/search/retriever/retrieval_server.py#L131) |
| `BaseRetriever.search` | `def(self, query, num, return_score)` | [`examples/search/retriever/retrieval_server.py`](../../examples/search/retriever/retrieval_server.py#L128) |
| `BaseRollout.__init__` | `def(self)` | [`verl/workers/rollout/base.py`](../../verl/workers/rollout/base.py#L23) |
| `BaseRollout.generate_sequences` | `def(self, prompts)` | [`verl/workers/rollout/base.py`](../../verl/workers/rollout/base.py#L33) |
| `BaseShardingManager.__enter__` | `def(self)` | [`verl/workers/sharding_manager/base.py`](../../verl/workers/sharding_manager/base.py#L22) |
| `BaseShardingManager.__exit__` | `def(self, exc_type, exc_value, traceback)` | [`verl/workers/sharding_manager/base.py`](../../verl/workers/sharding_manager/base.py#L25) |
| `BaseShardingManager.postprocess_data` | `def(self, data)` | [`verl/workers/sharding_manager/base.py`](../../verl/workers/sharding_manager/base.py#L31) |
| `BaseShardingManager.preprocess_data` | `def(self, data)` | [`verl/workers/sharding_manager/base.py`](../../verl/workers/sharding_manager/base.py#L28) |
| `BaseTool.__init__` | `def(self, config, tool_schema)` | [`verl/tools/base_tool.py`](../../verl/tools/base_tool.py#L33) |
| `BaseTool.calc_reward` | `async def(self, instance_id, **kwargs)` | [`verl/tools/base_tool.py`](../../verl/tools/base_tool.py#L69) |
| `BaseTool.create` | `async def(self, instance_id, **kwargs)` | [`verl/tools/base_tool.py`](../../verl/tools/base_tool.py#L41) |
| `BaseTool.execute` | `async def(self, instance_id, parameters, **kwargs)` | [`verl/tools/base_tool.py`](../../verl/tools/base_tool.py#L55) |
| `BaseTool.get_openai_tool_schema` | `def(self)` | [`verl/tools/base_tool.py`](../../verl/tools/base_tool.py#L38) |
| `BaseTool.release` | `async def(self, instance_id, **kwargs)` | [`verl/tools/base_tool.py`](../../verl/tools/base_tool.py#L80) |
| `BatchRewardManager.__call__` | `def(self, data, return_dict)` | [`verl/workers/reward_manager/batch.py`](../../verl/workers/reward_manager/batch.py#L59) |
| `BatchRewardManager.__init__` | `def(self, tokenizer, num_examine, compute_score, reward_fn_key, **reward_kwargs)` | [`verl/workers/reward_manager/batch.py`](../../verl/workers/reward_manager/batch.py#L23) |
| `BatchRewardManager.verify` | `def(self, data)` | [`verl/workers/reward_manager/batch.py`](../../verl/workers/reward_manager/batch.py#L30) |
| `bert_predict` | `def(obs, info, softmax)` | [`agent_system/environments/env_package/webshop/webshop/transfer/app.py`](../../agent_system/environments/env_package/webshop/webshop/transfer/app.py#L73) |
| `BertConfigForWebshop.__init__` | `def(self, pretrained_bert, image, **kwargs)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/models/bert.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/models/bert.py#L17) |
| `BertModelForWebshop.__init__` | `def(self, config)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/models/bert.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/models/bert.py#L34) |
| `BertModelForWebshop.forward` | `def(self, state_input_ids, state_attention_mask, action_input_ids, action_attention_mask, sizes, images, labels)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/models/bert.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/models/bert.py#L58) |
| `BertModelForWebshop.rl_forward` | `def(self, state_batch, act_batch, value, q, act)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/models/bert.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/models/bert.py#L86) |
| `BiAttention.__init__` | `def(self, input_size, dropout)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/models/modules.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/models/modules.py#L121) |
| `BiAttention.forward` | `def(self, context, memory, mask)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/models/modules.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/models/modules.py#L134) |
| `BiAttention.init_parameters` | `def(self)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/models/modules.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/models/modules.py#L131) |
| `BlackjackEnv.__init__` | `def(self, render_mode, natural, sab, is_pixel)` | [`agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/blackjack.py`](../../agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/blackjack.py#L181) |
| `BlackjackEnv._get_obs` | `def(self)` | [`agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/blackjack.py`](../../agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/blackjack.py#L265) |
| `BlackjackEnv.reset` | `def(self, seed, options)` | [`agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/blackjack.py`](../../agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/blackjack.py#L238) |
| `BlackjackEnv.step` | `def(self, action)` | [`agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/blackjack.py`](../../agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/blackjack.py#L201) |
| `BM25Retriever.__init__` | `def(self, config)` | [`examples/search/retriever/retrieval_server.py`](../../examples/search/retriever/retrieval_server.py#L136) |
| `BM25Retriever._batch_search` | `def(self, query_list, num, return_score)` | [`examples/search/retriever/retrieval_server.py`](../../examples/search/retriever/retrieval_server.py#L182) |
| `BM25Retriever._check_contain_doc` | `def(self)` | [`examples/search/retriever/retrieval_server.py`](../../examples/search/retriever/retrieval_server.py#L146) |
| `BM25Retriever._search` | `def(self, query, num, return_score)` | [`examples/search/retriever/retrieval_server.py`](../../examples/search/retriever/retrieval_server.py#L149) |
| `bootstrap_metric` | `def(data, subset_size, reduce_fns, n_bootstrap, seed)` | [`verl/trainer/ppo/metric_utils.py`](../../verl/trainer/ppo/metric_utils.py#L423) |
| `box_displacement_score` | `def(box_mapping)` | [`agent_system/environments/env_package/sokoban/sokoban/room_utils.py`](../../agent_system/environments/env_package/sokoban/sokoban/room_utils.py#L551) |
| `broadcast_dict_tensor` | `def(tensors, src, group)` | [`verl/utils/torch_functional.py`](../../verl/utils/torch_functional.py#L221) |
| `broadcast_from_megatron_pp` | `def(tensor)` | [`verl/utils/megatron_utils.py`](../../verl/utils/megatron_utils.py#L610) |
| `broadcast_pyobj` | `def(data, rank, dist_group, src, force_cpu_device)` | [`verl/workers/rollout/sglang_rollout/utils.py`](../../verl/workers/rollout/sglang_rollout/utils.py#L24) |
| `broadcast_str_from_megatron_pp` | `def(obj)` | [`verl/utils/megatron_utils.py`](../../verl/utils/megatron_utils.py#L645) |
| `build_aime2024_dataset` | `def()` | [`recipe/r1/data_process.py`](../../recipe/r1/data_process.py#L39) |
| `build_aime2024_dataset.process_aime2024` | `def(example)` | [`recipe/r1/data_process.py`](../../recipe/r1/data_process.py#L40) |
| `build_alfworld_envs` | `def(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train, env_kwargs)` | [`agent_system/environments/env_package/alfworld/envs.py`](../../agent_system/environments/env_package/alfworld/envs.py#L297) |
| `build_appworld_envs` | `def(dataset_name, max_interactions, seed, env_num, group_n, start_server_id, resources_per_worker)` | [`agent_system/environments/env_package/appworld/envs.py`](../../agent_system/environments/env_package/appworld/envs.py#L238) |
| `build_cnmo2024_dataset` | `def()` | [`recipe/r1/data_process.py`](../../recipe/r1/data_process.py#L74) |
| `build_cnmo2024_dataset.process_cnmo2024` | `def(example)` | [`recipe/r1/data_process.py`](../../recipe/r1/data_process.py#L75) |
| `build_env` | `def(env_name, env_num)` | [`examples/prompt_agent/gpt4o_alfworld.py`](../../examples/prompt_agent/gpt4o_alfworld.py#L9) |
| `build_gpqa_dimond_dataset` | `def()` | [`recipe/r1/data_process.py`](../../recipe/r1/data_process.py#L51) |
| `build_gpqa_dimond_dataset.process_gpqa_diamond` | `def(example)` | [`recipe/r1/data_process.py`](../../recipe/r1/data_process.py#L56) |
| `build_gymcards_envs` | `def(env_name, seed, env_num, group_n, resources_per_worker, is_train)` | [`agent_system/environments/env_package/gym_cards/envs.py`](../../agent_system/environments/env_package/gym_cards/envs.py#L163) |
| `build_livecodebench_dataset` | `def()` | [`recipe/r1/data_process.py`](../../recipe/r1/data_process.py#L93) |
| `build_livecodebench_dataset.process_livecodebench` | `def(example)` | [`recipe/r1/data_process.py`](../../recipe/r1/data_process.py#L99) |
| `build_memory_buffer` | `def(weight_buffer_meta)` | [`verl/utils/memory_buffer.py`](../../verl/utils/memory_buffer.py#L70) |
| `build_memory_reference` | `def(weight_buffer_meta, memory_buffers)` | [`verl/utils/memory_buffer.py`](../../verl/utils/memory_buffer.py#L113) |
| `build_memory_reference_from_module` | `def(module, memory_buffers, maintain_weight)` | [`verl/utils/memory_buffer.py`](../../verl/utils/memory_buffer.py#L99) |
| `build_search_envs` | `def(seed, env_num, group_n, is_train, env_config)` | [`agent_system/environments/env_package/search/envs.py`](../../agent_system/environments/env_package/search/envs.py#L162) |
| `build_sokoban_envs` | `def(seed, env_num, group_n, mode, resources_per_worker, is_train, env_kwargs)` | [`agent_system/environments/env_package/sokoban/envs.py`](../../agent_system/environments/env_package/sokoban/envs.py#L177) |
| `build_step_group` | `def(anchor_obs, index, enable_similarity, similarity_thresh, summarize)` | [`gigpo/core_gigpo.py`](../../gigpo/core_gigpo.py#L243) |
| `build_teacher_batch` | `def(batch, skill_provider, tokenizer, max_prompt_length, truncation)` | [`verl/trainer/ppo/rlsd_ray_trainer.py`](../../verl/trainer/ppo/rlsd_ray_trainer.py#L47) |
| `build_webshop_envs` | `def(seed, env_num, group_n, resources_per_worker, is_train, env_kwargs)` | [`agent_system/environments/env_package/webshop/envs.py`](../../agent_system/environments/env_package/webshop/envs.py#L261) |
| `calc_maj_val` | `def(data, vote_key, val_key)` | [`verl/trainer/ppo/metric_utils.py`](../../verl/trainer/ppo/metric_utils.py#L464) |
| `calc_padded_numel` | `def(shape, dtype)` | [`verl/utils/memory_buffer.py`](../../verl/utils/memory_buffer.py#L53) |
| `call_method` | `def(method, inputs)` | [`verl/utils/reward_score/prime_code/testing_util.py`](../../verl/utils/reward_score/prime_code/testing_util.py#L530) |
| `call_method._inner_call_method` | `def(_method)` | [`verl/utils/reward_score/prime_code/testing_util.py`](../../verl/utils/reward_score/prime_code/testing_util.py#L545) |
| `call_sandbox_api` | `def(sandbox_fusion_url, code, stdin, compile_timeout, run_timeout, language)` | [`verl/utils/reward_score/sandbox_fusion/utils.py`](../../verl/utils/reward_score/sandbox_fusion/utils.py#L37) |
| `call_search_api` | `def(retrieval_service_url, query, topk, return_scores, timeout, log_requests, session)` | [`agent_system/environments/env_package/search/third_party/skyrl_gym/tools/search.py`](../../agent_system/environments/env_package/search/third_party/skyrl_gym/tools/search.py#L42) |
| `call_search_api` | `def(retrieval_service_url, query_list, topk, return_scores, timeout)` | [`verl/tools/utils/search_r1_like_utils.py`](../../verl/tools/utils/search_r1_like_utils.py#L46) |
| `Capturing.__enter__` | `def(self)` | [`verl/utils/reward_score/prime_code/testing_util.py`](../../verl/utils/reward_score/prime_code/testing_util.py#L55) |
| `Capturing.__exit__` | `def(self, *args)` | [`verl/utils/reward_score/prime_code/testing_util.py`](../../verl/utils/reward_score/prime_code/testing_util.py#L62) |
| `ceildiv` | `def(a, b)` | [`verl/utils/seqlen_balancing.py`](../../verl/utils/seqlen_balancing.py#L221) |
| `CharTokenizer.__init__` | `def(self, characters, model_max_length, chat_template, **kwargs)` | [`tests/e2e/envs/digit_completion/tokenizer.py`](../../tests/e2e/envs/digit_completion/tokenizer.py#L30) |
| `CharTokenizer._convert_id_to_token` | `def(self, index)` | [`tests/e2e/envs/digit_completion/tokenizer.py`](../../tests/e2e/envs/digit_completion/tokenizer.py#L95) |
| `CharTokenizer._convert_token_to_id` | `def(self, token)` | [`tests/e2e/envs/digit_completion/tokenizer.py`](../../tests/e2e/envs/digit_completion/tokenizer.py#L92) |
| `CharTokenizer._tokenize` | `def(self, text)` | [`tests/e2e/envs/digit_completion/tokenizer.py`](../../tests/e2e/envs/digit_completion/tokenizer.py#L89) |
| `CharTokenizer.build_inputs_with_special_tokens` | `def(self, token_ids_0, token_ids_1)` | [`tests/e2e/envs/digit_completion/tokenizer.py`](../../tests/e2e/envs/digit_completion/tokenizer.py#L101) |
| `CharTokenizer.convert_tokens_to_string` | `def(self, tokens)` | [`tests/e2e/envs/digit_completion/tokenizer.py`](../../tests/e2e/envs/digit_completion/tokenizer.py#L98) |
| `CharTokenizer.from_config` | `def(cls, config)` | [`tests/e2e/envs/digit_completion/tokenizer.py`](../../tests/e2e/envs/digit_completion/tokenizer.py#L135) |
| `CharTokenizer.from_pretrained` | `def(cls, save_directory, **kwargs)` | [`tests/e2e/envs/digit_completion/tokenizer.py`](../../tests/e2e/envs/digit_completion/tokenizer.py#L149) |
| `CharTokenizer.get_config` | `def(self)` | [`tests/e2e/envs/digit_completion/tokenizer.py`](../../tests/e2e/envs/digit_completion/tokenizer.py#L127) |
| `CharTokenizer.get_special_tokens_mask` | `def(self, token_ids_0, token_ids_1, already_has_special_tokens)` | [`tests/e2e/envs/digit_completion/tokenizer.py`](../../tests/e2e/envs/digit_completion/tokenizer.py#L109) |
| `CharTokenizer.get_vocab` | `def(self)` | [`tests/e2e/envs/digit_completion/tokenizer.py`](../../tests/e2e/envs/digit_completion/tokenizer.py#L86) |
| `CharTokenizer.save_pretrained` | `def(self, save_directory, **kwargs)` | [`tests/e2e/envs/digit_completion/tokenizer.py`](../../tests/e2e/envs/digit_completion/tokenizer.py#L142) |
| `CharTokenizer.vocab_size` | `def(self)` | [`tests/e2e/envs/digit_completion/tokenizer.py`](../../tests/e2e/envs/digit_completion/tokenizer.py#L83) |
| `ChatCompletionScheduler.__init__` | `def(self, config, model_path, server_addresses, max_cache_size)` | [`verl/workers/rollout/async_server.py`](../../verl/workers/rollout/async_server.py#L109) |
| `ChatCompletionScheduler._chat_completions_aiohttp` | `async def(self, address, **chat_complete_request)` | [`verl/workers/rollout/async_server.py`](../../verl/workers/rollout/async_server.py#L199) |
| `ChatCompletionScheduler._chat_completions_openai` | `async def(self, address, **chat_complete_request)` | [`verl/workers/rollout/async_server.py`](../../verl/workers/rollout/async_server.py#L195) |
| `ChatCompletionScheduler.generate_sequences` | `async def(self, prompts, **sampling_params)` | [`verl/workers/rollout/async_server.py`](../../verl/workers/rollout/async_server.py#L214) |
| `ChatCompletionScheduler.submit_chat_completions` | `async def(self, callback, callback_additional_info, **chat_complete_request)` | [`verl/workers/rollout/async_server.py`](../../verl/workers/rollout/async_server.py#L135) |
| `check_and_construct_configs` | `def(original_config, cls)` | [`verl/models/mcore/config_converter.py`](../../verl/models/mcore/config_converter.py#L137) |
| `check_congratulations_in_file` | `def(output_file)` | [`tests/e2e/check_custom_rwd_fn.py`](../../tests/e2e/check_custom_rwd_fn.py#L18) |
| `check_correctness` | `def(in_outs, generation, timeout, debug)` | [`recipe/r1/tasks/livecodebench.py`](../../recipe/r1/tasks/livecodebench.py#L31) |
| `check_correctness` | `def(in_outs, generation, timeout, debug)` | [`verl/utils/reward_score/prime_code/utils.py`](../../verl/utils/reward_score/prime_code/utils.py#L41) |
| `check_correctness` | `def(sandbox_fusion_url, in_outs, generation, timeout, language, concurrent_semaphore)` | [`verl/utils/reward_score/sandbox_fusion/utils.py`](../../verl/utils/reward_score/sandbox_fusion/utils.py#L367) |
| `check_cuda_is_available` | `def()` | [`verl/utils/torch_functional.py`](../../verl/utils/torch_functional.py#L645) |
| `check_cuda_versions` | `def()` | [`scripts/diagnose.py`](../../scripts/diagnose.py#L187) |
| `check_environment` | `def()` | [`scripts/diagnose.py`](../../scripts/diagnose.py#L170) |
| `check_expected_config` | `def(config, expect_file)` | [`verl/utils/expected_config.py`](../../verl/utils/expected_config.py#L97) |
| `check_hardware` | `def()` | [`scripts/diagnose.py`](../../scripts/diagnose.py#L135) |
| `check_network` | `def(args)` | [`scripts/diagnose.py`](../../scripts/diagnose.py#L151) |
| `check_os` | `def()` | [`scripts/diagnose.py`](../../scripts/diagnose.py#L126) |
| `check_pip` | `def()` | [`scripts/diagnose.py`](../../scripts/diagnose.py#L78) |
| `check_pip_package_versions` | `def()` | [`scripts/diagnose.py`](../../scripts/diagnose.py#L177) |
| `check_python` | `def()` | [`scripts/diagnose.py`](../../scripts/diagnose.py#L70) |
| `check_system_info` | `def()` | [`scripts/diagnose.py`](../../scripts/diagnose.py#L253) |
| `check_verl` | `def()` | [`scripts/diagnose.py`](../../scripts/diagnose.py#L101) |
| `check_workers_alive` | `def(workers, is_alive, gap_time)` | [`verl/single_controller/base/worker_group.py`](../../verl/single_controller/base/worker_group.py#L100) |
| `ClassWithInitArgs.__call__` | `def(self)` | [`verl/single_controller/base/worker_group.py`](../../verl/single_controller/base/worker_group.py#L95) |
| `ClassWithInitArgs.__init__` | `def(self, cls, *args, **kwargs)` | [`verl/single_controller/base/worker_group.py`](../../verl/single_controller/base/worker_group.py#L81) |
| `clean_product_keys` | `def(products)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/engine.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/engine.py#L210) |
| `clean_torchelastic_env` | `def()` | [`tests/workers/rollout/utils_sglang.py`](../../tests/workers/rollout/utils_sglang.py#L82) |
| `clean_traceback` | `def(error_traceback)` | [`verl/utils/reward_score/prime_code/testing_util.py`](../../verl/utils/reward_score/prime_code/testing_util.py#L80) |
| `ClearMLLogger.__init__` | `def(self, project_name, experiment_name, config)` | [`verl/utils/tracking.py`](../../verl/utils/tracking.py#L147) |
| `ClearMLLogger._get_logger` | `def(self)` | [`verl/utils/tracking.py`](../../verl/utils/tracking.py#L162) |
| `ClearMLLogger.finish` | `def(self)` | [`verl/utils/tracking.py`](../../verl/utils/tracking.py#L191) |
| `ClearMLLogger.log` | `def(self, data, step)` | [`verl/utils/tracking.py`](../../verl/utils/tracking.py#L165) |
| `clip_by_value` | `def(x, tensor_min, tensor_max)` | [`verl/utils/torch_functional.py`](../../verl/utils/torch_functional.py#L115) |
| `cmp` | `def(a, b)` | [`agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/blackjack.py`](../../agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/blackjack.py#L36) |
| `collate_fn` | `def(x)` | [`verl/protocol.py`](../../verl/protocol.py#L179) |
| `collate_fn` | `def(data_list)` | [`verl/utils/dataset/rl_dataset.py`](../../verl/utils/dataset/rl_dataset.py#L46) |
| `collect_all_to_all` | `def(worker_group, output)` | [`verl/single_controller/base/decorator.py`](../../verl/single_controller/base/decorator.py#L146) |
| `collect_async` | `async def(n_traj, max_steps, initial_obs, generate_action, env_step, rollout_n, uid_factory, traj_uid_factory, max_in_flight)` | [`agent_system/multi_turn_rollout/async_rollout_core.py`](../../agent_system/multi_turn_rollout/async_rollout_core.py#L111) |
| `collect_async._generate` | `async def(traj_id, step, obs)` | [`agent_system/multi_turn_rollout/async_rollout_core.py`](../../agent_system/multi_turn_rollout/async_rollout_core.py#L152) |
| `collect_async._run_trajectory` | `async def(i)` | [`agent_system/multi_turn_rollout/async_rollout_core.py`](../../agent_system/multi_turn_rollout/async_rollout_core.py#L164) |
| `collect_dp_compute` | `def(worker_group, output)` | [`verl/single_controller/base/decorator.py`](../../verl/single_controller/base/decorator.py#L355) |
| `collect_dp_compute_data_proto` | `def(worker_group, output)` | [`verl/single_controller/base/decorator.py`](../../verl/single_controller/base/decorator.py#L387) |
| `collect_megatron_compute` | `def(worker_group, output)` | [`verl/single_controller/base/decorator.py`](../../verl/single_controller/base/decorator.py#L179) |
| `collect_megatron_compute_data_proto` | `def(worker_group, output)` | [`verl/single_controller/base/decorator.py`](../../verl/single_controller/base/decorator.py#L226) |
| `collect_megatron_pp_as_dp` | `def(worker_group, output)` | [`verl/single_controller/base/decorator.py`](../../verl/single_controller/base/decorator.py#L294) |
| `collect_megatron_pp_as_dp_data_proto` | `def(worker_group, output)` | [`verl/single_controller/base/decorator.py`](../../verl/single_controller/base/decorator.py#L335) |
| `collect_megatron_pp_only` | `def(worker_group, output)` | [`verl/single_controller/base/decorator.py`](../../verl/single_controller/base/decorator.py#L309) |
| `combined_int_check` | `def(val)` | [`verl/utils/reward_score/prime_code/testing_util.py`](../../verl/utils/reward_score/prime_code/testing_util.py#L76) |
| `compute_advantage` | `def(data, adv_estimator, gamma, lam, num_repeat, multi_turn, norm_adv_by_std_in_grpo, step_advantage_w, gigpo_mode, **kwargs)` | [`recipe/hgpo/hgpo_ray_trainer.py`](../../recipe/hgpo/hgpo_ray_trainer.py#L247) |
| `compute_advantage` | `def(data, adv_estimator, config)` | [`recipe/prime/prime_ray_trainer.py`](../../recipe/prime/prime_ray_trainer.py#L41) |
| `compute_advantage` | `def(data, beta)` | [`recipe/sppo/sppo_ray_trainer.py`](../../recipe/sppo/sppo_ray_trainer.py#L60) |
| `compute_advantage` | `def(data, adv_estimator, gamma, lam, num_repeat, multi_turn, norm_adv_by_std_in_grpo, step_advantage_w, gigpo_mode, gigpo_enable_similarity, gigpo_similarity_thresh, **kwargs)` | [`verl/trainer/ppo/ray_trainer.py`](../../verl/trainer/ppo/ray_trainer.py#L283) |
| `compute_attention_mask` | `def(prompts, pad_token_id)` | [`tests/e2e/envs/digit_completion/task.py`](../../tests/e2e/envs/digit_completion/task.py#L105) |
| `compute_ce_dpo_loss_rm` | `def(token_level_scores, acc, response_mask, beta)` | [`recipe/prime/prime_core_algos.py`](../../recipe/prime/prime_core_algos.py#L75) |
| `compute_data_metrics` | `def(batch, use_critic)` | [`recipe/prime/prime_ray_trainer.py`](../../recipe/prime/prime_ray_trainer.py#L55) |
| `compute_data_metrics` | `def(batch, use_critic)` | [`verl/trainer/ppo/metric_utils.py`](../../verl/trainer/ppo/metric_utils.py#L212) |
| `compute_data_metrics_by_task` | `def(batch, use_critic)` | [`verl/trainer/ppo/metric_utils.py`](../../verl/trainer/ppo/metric_utils.py#L157) |
| `compute_detach_dpo_loss_rm` | `def(token_level_scores, acc, Q_bc, acc_bc, response_mask, beta, bon_mode)` | [`recipe/prime/prime_core_algos.py`](../../recipe/prime/prime_core_algos.py#L81) |
| `compute_dpo_abs_accuracy` | `def(token_level_scores, acc, response_mask, n_samples)` | [`recipe/prime/prime_core_algos.py`](../../recipe/prime/prime_core_algos.py#L135) |
| `compute_dpo_accuracy` | `def(token_level_scores, acc, response_mask, n_samples)` | [`recipe/prime/prime_core_algos.py`](../../recipe/prime/prime_core_algos.py#L112) |
| `compute_dpo_accuracy.get_upper_triangle` | `def(tensor_x)` | [`recipe/prime/prime_core_algos.py`](../../recipe/prime/prime_core_algos.py#L117) |
| `compute_dpo_data_metrics` | `def(batch)` | [`recipe/spin/spin_trainer.py`](../../recipe/spin/spin_trainer.py#L187) |
| `compute_entropy_loss` | `def(logits, response_mask, loss_agg_mode)` | [`verl/trainer/ppo/core_algos.py`](../../verl/trainer/ppo/core_algos.py#L589) |
| `compute_gae_advantage_return` | `def(token_level_rewards, values, response_mask, gamma, lam)` | [`verl/trainer/ppo/core_algos.py`](../../verl/trainer/ppo/core_algos.py#L67) |
| `compute_gigpo_outcome_advantage` | `def(token_level_rewards, step_rewards, response_mask, anchor_obs, index, traj_index, epsilon, step_advantage_w, mode, enable_similarity, similarity_thresh)` | [`gigpo/core_gigpo.py`](../../gigpo/core_gigpo.py#L138) |
| `compute_grad_norm` | `def(model)` | [`verl/utils/torch_functional.py`](../../verl/utils/torch_functional.py#L213) |
| `compute_grpo_outcome_advantage` | `def(token_level_rewards, response_mask, index, traj_index, epsilon, norm_adv_by_std_in_grpo, compute_mean_std_cross_steps)` | [`verl/trainer/ppo/core_algos.py`](../../verl/trainer/ppo/core_algos.py#L113) |
| `compute_grpo_passk_outcome_advantage` | `def(token_level_rewards, response_mask, index, traj_index, epsilon, norm_adv_by_std_in_grpo, compute_mean_std_cross_steps)` | [`verl/trainer/ppo/core_algos.py`](../../verl/trainer/ppo/core_algos.py#L177) |
| `compute_hgpo_outcome_advantage` | `def(token_level_rewards, step_rewards, response_mask, anchor_obs, index, traj_index, history_length, epsilon, mode, length_weight_alpha, base_group)` | [`recipe/hgpo/core_hgpo.py`](../../recipe/hgpo/core_hgpo.py#L66) |
| `compute_log_prob_with_prefetch` | `def(actor_rollout_wg, batch, prefetched, temperature)` | [`agent_system/multi_turn_rollout/utils.py`](../../agent_system/multi_turn_rollout/utils.py#L87) |
| `compute_metrics_by_task` | `def(batch, metric_fn)` | [`verl/trainer/ppo/metric_utils.py`](../../verl/trainer/ppo/metric_utils.py#L133) |
| `compute_online_dpo_loss` | `def(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta, label_smoothing, loss_type, reference_free)` | [`recipe/spin/core_algos.py`](../../recipe/spin/core_algos.py#L131) |
| `compute_onlinedpo_pref` | `def(token_level_rewards, response_mask)` | [`recipe/spin/core_algos.py`](../../recipe/spin/core_algos.py#L60) |
| `compute_onlineDPO_pref` | `def(data)` | [`recipe/spin/spin_trainer.py`](../../recipe/spin/spin_trainer.py#L301) |
| `compute_opd_data_metrics` | `def(batch)` | [`verl/trainer/ppo/opd_ray_trainer.py`](../../verl/trainer/ppo/opd_ray_trainer.py#L56) |
| `compute_opd_data_metrics_by_task` | `def(batch)` | [`verl/trainer/ppo/opd_ray_trainer.py`](../../verl/trainer/ppo/opd_ray_trainer.py#L104) |
| `compute_pf_ppo_reweight_data` | `def(data, reweight_method, weight_pow)` | [`verl/trainer/ppo/core_algos.py`](../../verl/trainer/ppo/core_algos.py#L717) |
| `compute_pf_ppo_reweight_data.compute_weights` | `def(scores, reweight_method, weight_pow)` | [`verl/trainer/ppo/core_algos.py`](../../verl/trainer/ppo/core_algos.py#L734) |
| `compute_policy_loss` | `def(old_log_prob, log_prob, advantages, response_mask, cliprange, cliprange_low, cliprange_high, clip_ratio_c, loss_agg_mode)` | [`verl/trainer/ppo/core_algos.py`](../../verl/trainer/ppo/core_algos.py#L457) |
| `compute_policy_loss_gspo` | `def(old_log_prob, log_prob, advantages, response_mask, cliprange, cliprange_low, cliprange_high, clip_ratio_c, loss_agg_mode)` | [`verl/trainer/ppo/core_algos.py`](../../verl/trainer/ppo/core_algos.py#L521) |
| `compute_position_id_with_mask` | `def(mask)` | [`tests/e2e/envs/digit_completion/task.py`](../../tests/e2e/envs/digit_completion/task.py#L111) |
| `compute_position_id_with_mask` | `def(mask)` | [`tests/ray_gpu/detached_worker/client.py`](../../tests/ray_gpu/detached_worker/client.py#L28) |
| `compute_position_id_with_mask` | `def(mask)` | [`verl/utils/model.py`](../../verl/utils/model.py#L204) |
| `compute_reinforce_plus_plus_baseline_outcome_advantage` | `def(token_level_rewards, response_mask, index, traj_index, epsilon, compute_mean_std_cross_steps)` | [`verl/trainer/ppo/core_algos.py`](../../verl/trainer/ppo/core_algos.py#L239) |
| `compute_reinforce_plus_plus_outcome_advantage` | `def(token_level_rewards, response_mask, gamma)` | [`verl/trainer/ppo/core_algos.py`](../../verl/trainer/ppo/core_algos.py#L329) |
| `compute_remax_outcome_advantage` | `def(token_level_rewards, reward_baselines, response_mask)` | [`verl/trainer/ppo/core_algos.py`](../../verl/trainer/ppo/core_algos.py#L362) |
| `compute_response_mask` | `def(data)` | [`recipe/hgpo/hgpo_ray_trainer.py`](../../recipe/hgpo/hgpo_ray_trainer.py#L229) |
| `compute_response_mask` | `def(data)` | [`recipe/prime/prime_ray_trainer.py`](../../recipe/prime/prime_ray_trainer.py#L114) |
| `compute_response_mask` | `def(data)` | [`recipe/spin/spin_trainer.py`](../../recipe/spin/spin_trainer.py#L294) |
| `compute_response_mask` | `def(data)` | [`verl/trainer/ppo/ray_trainer.py`](../../verl/trainer/ppo/ray_trainer.py#L265) |
| `compute_reward` | `def(info, multi_modal)` | [`agent_system/environments/env_package/alfworld/envs.py`](../../agent_system/environments/env_package/alfworld/envs.py#L48) |
| `compute_reward` | `def(prompt, response, sequence_reward)` | [`tests/e2e/envs/digit_completion/task.py`](../../tests/e2e/envs/digit_completion/task.py#L135) |
| `compute_reward` | `def(data, reward_fn)` | [`verl/trainer/ppo/reward.py`](../../verl/trainer/ppo/reward.py#L103) |
| `compute_reward_async` | `def(data, config, tokenizer)` | [`verl/trainer/ppo/reward.py`](../../verl/trainer/ppo/reward.py#L125) |
| `compute_rewards` | `def(token_level_scores, old_log_prob, ref_log_prob, kl_ratio)` | [`verl/trainer/ppo/core_algos.py`](../../verl/trainer/ppo/core_algos.py#L390) |
| `compute_rloo_advantage_return` | `def(data, response_mask, n_samples, config)` | [`recipe/prime/prime_core_algos.py`](../../recipe/prime/prime_core_algos.py#L21) |
| `compute_rloo_advantage_return.masked_rloo` | `def(reward_tensor_original, mask_tensor)` | [`recipe/prime/prime_core_algos.py`](../../recipe/prime/prime_core_algos.py#L23) |
| `compute_rloo_outcome_advantage` | `def(token_level_rewards, response_mask, index, traj_index, epsilon, compute_mean_std_cross_steps)` | [`verl/trainer/ppo/core_algos.py`](../../verl/trainer/ppo/core_algos.py#L285) |
| `compute_rlsd_token_advantage` | `def(seq_advantages, student_log_probs, teacher_log_probs, response_mask, rlsd_lambda, rlsd_clip_eps)` | [`verl/trainer/ppo/rlsd_utils.py`](../../verl/trainer/ppo/rlsd_utils.py#L211) |
| `compute_score` | `def(solution_str, ground_truth)` | [`recipe/r1/tasks/gpqa.py`](../../recipe/r1/tasks/gpqa.py#L21) |
| `compute_score` | `def(completion, test_cases)` | [`recipe/r1/tasks/livecodebench.py`](../../recipe/r1/tasks/livecodebench.py#L55) |
| `compute_score` | `def(model_output, ground_truth)` | [`recipe/r1/tasks/math.py`](../../recipe/r1/tasks/math.py#L23) |
| `compute_score` | `def(predict_str, ground_truth)` | [`verl/utils/reward_score/geo3k.py`](../../verl/utils/reward_score/geo3k.py#L31) |
| `compute_score` | `def(solution_str, ground_truth, method, format_score, score)` | [`verl/utils/reward_score/gsm8k.py`](../../verl/utils/reward_score/gsm8k.py#L44) |
| `compute_score` | `def(solution_str, ground_truth)` | [`verl/utils/reward_score/math.py`](../../verl/utils/reward_score/math.py#L17) |
| `compute_score` | `def(solution_str, ground_truth, strict_box_verify, pause_tokens_index)` | [`verl/utils/reward_score/math_dapo.py`](../../verl/utils/reward_score/math_dapo.py#L237) |
| `compute_score` | `def(model_output, ground_truth, timeout_score)` | [`verl/utils/reward_score/math_verify.py`](../../verl/utils/reward_score/math_verify.py#L23) |
| `compute_score` | `def(completion, test_cases, continuous)` | [`verl/utils/reward_score/prime_code/__init__.py`](../../verl/utils/reward_score/prime_code/__init__.py#L21) |
| `compute_score` | `def(model_output, ground_truth)` | [`verl/utils/reward_score/prime_math/__init__.py`](../../verl/utils/reward_score/prime_math/__init__.py#L379) |
| `compute_score` | `def(sandbox_fusion_url, concurrent_semaphore, completion, test_cases, continuous, timeout)` | [`verl/utils/reward_score/sandbox_fusion/__init__.py`](../../verl/utils/reward_score/sandbox_fusion/__init__.py#L28) |
| `compute_score` | `def(solution_str, ground_truth, method, format_score, score)` | [`verl/utils/reward_score/search_r1_like_qa_em.py`](../../verl/utils/reward_score/search_r1_like_qa_em.py#L96) |
| `compute_score_batched` | `def(data_sources, solution_strs, ground_truths, extra_infos)` | [`verl/utils/reward_score/math_batch.py`](../../verl/utils/reward_score/math_batch.py#L18) |
| `compute_score_subem` | `def(solution_str, ground_truth, method, format_score, score)` | [`verl/utils/reward_score/search_r1_like_qa_em.py`](../../verl/utils/reward_score/search_r1_like_qa_em.py#L131) |
| `compute_sdar_loss` | `def(student_log_probs, teacher_log_probs, response_mask, gate_beta, loss_agg_mode)` | [`verl/trainer/ppo/sdar_utils.py`](../../verl/trainer/ppo/sdar_utils.py#L14) |
| `compute_sdl_loss` | `def(student_log_probs, teacher_log_probs, old_log_probs, response_mask, loss_agg_mode)` | [`verl/trainer/ppo/skillsd_utils.py`](../../verl/trainer/ppo/skillsd_utils.py#L14) |
| `compute_sppo_loss` | `def(old_log_prob, log_prob, rewards, response_mask, eta, loss_agg_mode)` | [`recipe/sppo/dp_actor.py`](../../recipe/sppo/dp_actor.py#L33) |
| `compute_step_discounted_returns` | `def(batch, gamma)` | [`gigpo/core_gigpo.py`](../../gigpo/core_gigpo.py#L87) |
| `compute_step_discounted_returns` | `def(batch, gamma)` | [`recipe/hgpo/core_hgpo.py`](../../recipe/hgpo/core_hgpo.py#L27) |
| `compute_throughout_metrics` | `def(batch, timing_raw, n_gpus)` | [`verl/trainer/ppo/metric_utils.py`](../../verl/trainer/ppo/metric_utils.py#L375) |
| `compute_timing_metrics` | `def(batch, timing_raw)` | [`recipe/prime/prime_ray_trainer.py`](../../recipe/prime/prime_ray_trainer.py#L121) |
| `compute_timing_metrics` | `def(batch, timing_raw)` | [`verl/trainer/ppo/metric_utils.py`](../../verl/trainer/ppo/metric_utils.py#L336) |
| `compute_trajectory_response_tokens` | `def(batch)` | [`verl/trainer/ppo/metric_utils.py`](../../verl/trainer/ppo/metric_utils.py#L195) |
| `compute_transformers_input_shapes` | `def(batches, meta_info)` | [`verl/utils/megatron/pipeline_parallel.py`](../../verl/utils/megatron/pipeline_parallel.py#L22) |
| `compute_value_loss` | `def(vpreds, returns, values, response_mask, cliprange_value, loss_agg_mode)` | [`verl/trainer/ppo/core_algos.py`](../../verl/trainer/ppo/core_algos.py#L606) |
| `concat_dict_to_str` | `def(dict, step)` | [`verl/utils/logger/aggregate_logger.py`](../../verl/utils/logger/aggregate_logger.py#L23) |
| `Config.__init__` | `def(self, retrieval_method, retrieval_topk, index_path, corpus_path, dataset_path, data_split, faiss_gpu, retrieval_model_path, retrieval_pooling_method, retrieval_query_max_length, retrieval_use_fp16, retrieval_batch_size)` | [`examples/search/retriever/retrieval_server.py`](../../examples/search/retriever/retrieval_server.py#L275) |
| `Config.__init__` | `def(self)` | [`scripts/converter_hf_to_mcore.py`](../../scripts/converter_hf_to_mcore.py#L54) |
| `Config.__init__` | `def(self, config_dict)` | [`tests/utils/gpu_tests/test_flops_counter.py`](../../tests/utils/gpu_tests/test_flops_counter.py#L25) |
| `configure` | `def(dir, format_strs)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L392) |
| `configure_logger` | `def(log_dir, wandb)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/train_rl.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/train_rl.py#L14) |
| `convert_checkpoint_from_transformers_to_megatron` | `def(hf_model, model, hf_config)` | [`scripts/converter_hf_to_mcore.py`](../../scripts/converter_hf_to_mcore.py#L98) |
| `convert_checkpoint_from_transformers_to_megatron_dpskv3` | `def(hf_model, model, hf_config, tfconfig)` | [`scripts/converter_hf_to_mcore.py`](../../scripts/converter_hf_to_mcore.py#L150) |
| `convert_checkpoint_from_transformers_to_megatron_dpskv3.safe_copy` | `def(src_tensor, dst_tensor, skip_dtype_assert)` | [`scripts/converter_hf_to_mcore.py`](../../scripts/converter_hf_to_mcore.py#L153) |
| `convert_config` | `def(hf_config, megatron_config)` | [`verl/utils/megatron_utils.py`](../../verl/utils/megatron_utils.py#L159) |
| `convert_dict_to_actions` | `def(page_type, products, asin, page_num)` | [`agent_system/environments/env_package/webshop/webshop/transfer/predict_help.py`](../../agent_system/environments/env_package/webshop/webshop/transfer/predict_help.py#L429) |
| `convert_hf_to_mcore` | `def(hf_model_path, output_path, use_cpu_initialization, test, trust_remote_code)` | [`scripts/converter_hf_to_mcore.py`](../../scripts/converter_hf_to_mcore.py#L214) |
| `convert_hf_to_mcore.megatron_model_provider` | `def(pre_process, post_process)` | [`scripts/converter_hf_to_mcore.py`](../../scripts/converter_hf_to_mcore.py#L245) |
| `convert_html_to_text` | `def(html, simple, clicked_options, visited_asins)` | [`agent_system/environments/env_package/webshop/webshop/transfer/predict_help.py`](../../agent_system/environments/env_package/webshop/webshop/transfer/predict_help.py#L394) |
| `convert_html_to_text.tag_visible` | `def(element)` | [`agent_system/environments/env_package/webshop/webshop/transfer/predict_help.py`](../../agent_system/environments/env_package/webshop/webshop/transfer/predict_help.py#L395) |
| `convert_megatron_model_to_transformers_model` | `def(name, param, config, tp_size, num_query_groups, convert_qkv_gate_up_by_trunk_concat)` | [`verl/utils/megatron_utils.py`](../../verl/utils/megatron_utils.py#L470) |
| `convert_megatron_model_to_transformers_model.convert_gate_up_shard` | `def(full_tensor, gate_name, up_name)` | [`verl/utils/megatron_utils.py`](../../verl/utils/megatron_utils.py#L529) |
| `convert_megatron_model_to_transformers_model.convert_qkv_shard` | `def(full_tensor, q_name, k_name, v_name)` | [`verl/utils/megatron_utils.py`](../../verl/utils/megatron_utils.py#L481) |
| `convert_to_regular_types` | `def(obj)` | [`verl/utils/py_functional.py`](../../verl/utils/py_functional.py#L269) |
| `convert_web_app_string_to_var` | `def(name, string)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/engine.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/engine.py#L131) |
| `copy` | `def(src, dst, **kwargs)` | [`verl/utils/hdfs_io.py`](../../verl/utils/hdfs_io.py#L84) |
| `copy_local_path_from_hdfs` | `def(src, cache_dir, filelock, verbose, always_recopy)` | [`verl/utils/fs.py`](../../verl/utils/fs.py#L212) |
| `copy_to_local` | `def(src, cache_dir, filelock, verbose, always_recopy, use_shm)` | [`verl/utils/fs.py`](../../verl/utils/fs.py#L191) |
| `copy_to_shm` | `def(src)` | [`verl/utils/fs.py`](../../verl/utils/fs.py#L141) |
| `count_answer_tags` | `def(text)` | [`verl/utils/reward_score/search_r1_like_qa_em.py`](../../verl/utils/reward_score/search_r1_like_qa_em.py#L89) |
| `count_unknown_letters_in_expr` | `def(expr)` | [`verl/utils/reward_score/prime_math/__init__.py`](../../verl/utils/reward_score/prime_math/__init__.py#L193) |
| `CpuOffloadHookWithOffloadHandler.__enter__` | `def(self)` | [`verl/utils/activation_offload.py`](../../verl/utils/activation_offload.py#L72) |
| `CpuOffloadHookWithOffloadHandler.__exit__` | `def(self, *args)` | [`verl/utils/activation_offload.py`](../../verl/utils/activation_offload.py#L76) |
| `CpuOffloadHookWithOffloadHandler.__init__` | `def(self, offload_handler, handler_extra_kwargs)` | [`verl/utils/activation_offload.py`](../../verl/utils/activation_offload.py#L61) |
| `CpuOffloadHookWithOffloadHandler.on_get_saved_tensor` | `def(self, saved_state)` | [`verl/utils/activation_offload.py`](../../verl/utils/activation_offload.py#L84) |
| `CpuOffloadHookWithOffloadHandler.on_save_for_backward` | `def(self, tensor)` | [`verl/utils/activation_offload.py`](../../verl/utils/activation_offload.py#L80) |
| `create_colocated_worker_cls` | `def(class_dict)` | [`verl/single_controller/ray/base.py`](../../verl/single_controller/ray/base.py#L692) |
| `create_colocated_worker_cls.WorkerDict.__init__` | `def(self)` | [`verl/single_controller/ray/base.py`](../../verl/single_controller/ray/base.py#L711) |
| `create_colocated_worker_cls_fused` | `def(class_dict)` | [`verl/single_controller/ray/base.py`](../../verl/single_controller/ray/base.py#L799) |
| `create_colocated_worker_raw_cls` | `def(class_dict)` | [`verl/single_controller/ray/base.py`](../../verl/single_controller/ray/base.py#L735) |
| `create_colocated_worker_raw_cls.FusedWorker.__init__` | `def(self, *args, **kwargs)` | [`verl/single_controller/ray/base.py`](../../verl/single_controller/ray/base.py#L762) |
| `create_colocated_worker_raw_cls.FusedWorker._fuw_execute` | `def(self, method_name, *args, **kwargs)` | [`verl/single_controller/ray/base.py`](../../verl/single_controller/ray/base.py#L781) |
| `create_device_mesh` | `def(world_size, fsdp_size)` | [`recipe/spin/fsdp_workers.py`](../../recipe/spin/fsdp_workers.py#L47) |
| `create_device_mesh` | `def(world_size, fsdp_size)` | [`verl/workers/fsdp_workers.py`](../../verl/workers/fsdp_workers.py#L80) |
| `create_huggingface_actor` | `def(model_name, override_config_kwargs, automodel_kwargs)` | [`verl/utils/model.py`](../../verl/utils/model.py#L89) |
| `create_huggingface_critic` | `def(model_name, override_config_kwargs, automodel_kwargs)` | [`verl/utils/model.py`](../../verl/utils/model.py#L109) |
| `create_nccl_communicator_in_ray` | `def(rank, world_size, group_name, max_retries, interval_s)` | [`verl/utils/rendezvous/ray_backend.py`](../../verl/utils/rendezvous/ray_backend.py#L45) |
| `create_random_mask` | `def(input_ids, max_ratio_of_valid_token, max_ratio_of_left_padding, min_ratio_of_valid_token)` | [`verl/utils/model.py`](../../verl/utils/model.py#L161) |
| `create_rl_dataset` | `def(data_paths, data_config, tokenizer, processor)` | [`recipe/hgpo/main_hgpo.py`](../../recipe/hgpo/main_hgpo.py#L190) |
| `create_rl_dataset` | `def(data_paths, data_config, tokenizer, processor)` | [`verl/trainer/main_ppo.py`](../../verl/trainer/main_ppo.py#L191) |
| `create_rl_sampler` | `def(data_config, dataset)` | [`recipe/hgpo/main_hgpo.py`](../../recipe/hgpo/main_hgpo.py#L225) |
| `create_rl_sampler` | `def(data_config, dataset)` | [`verl/trainer/main_ppo.py`](../../verl/trainer/main_ppo.py#L309) |
| `create_sft_dataset` | `def(data_paths, data_config, tokenizer)` | [`verl/trainer/fsdp_sft_trainer.py`](../../verl/trainer/fsdp_sft_trainer.py#L586) |
| `create_trainer` | `def(config)` | [`tests/e2e/sft/test_sp_loss_match.py`](../../tests/e2e/sft/test_sp_loss_match.py#L90) |
| `create_worker_group_register_center` | `def(name, info)` | [`verl/single_controller/base/register_center/ray.py`](../../verl/single_controller/base/register_center/ray.py#L37) |
| `Critic.__init__` | `def(self, val)` | [`tests/ray_cpu/test_fused_workers.py`](../../tests/ray_cpu/test_fused_workers.py#L35) |
| `Critic.__init__` | `def(self, config)` | [`tests/ray_gpu/test_colocated_workers.py`](../../tests/ray_gpu/test_colocated_workers.py#L41) |
| `Critic.__init__` | `def(self, config)` | [`tests/ray_gpu/test_colocated_workers_fused.py`](../../tests/ray_gpu/test_colocated_workers_fused.py#L41) |
| `Critic.sub` | `def(self, x)` | [`tests/ray_cpu/test_fused_workers.py`](../../tests/ray_cpu/test_fused_workers.py#L40) |
| `Critic.sub` | `async def(self, data)` | [`tests/ray_gpu/test_colocated_workers.py`](../../tests/ray_gpu/test_colocated_workers.py#L46) |
| `Critic.sub` | `def(self, data)` | [`tests/ray_gpu/test_colocated_workers_fused.py`](../../tests/ray_gpu/test_colocated_workers_fused.py#L46) |
| `CriticWorker.__init__` | `def(self, config)` | [`verl/workers/fsdp_workers.py`](../../verl/workers/fsdp_workers.py#L878) |
| `CriticWorker.__init__` | `def(self, config)` | [`verl/workers/megatron_workers.py`](../../verl/workers/megatron_workers.py#L541) |
| `CriticWorker._build_critic_model_optimizer` | `def(self, config)` | [`verl/workers/fsdp_workers.py`](../../verl/workers/fsdp_workers.py#L919) |
| `CriticWorker._build_critic_model_optimizer` | `def(self, model_path, optim_config, override_model_config, override_transformer_config)` | [`verl/workers/megatron_workers.py`](../../verl/workers/megatron_workers.py#L585) |
| `CriticWorker._build_critic_model_optimizer.megatron_critic_model_provider` | `def(pre_process, post_process)` | [`verl/workers/megatron_workers.py`](../../verl/workers/megatron_workers.py#L594) |
| `CriticWorker.compute_values` | `def(self, data)` | [`verl/workers/fsdp_workers.py`](../../verl/workers/fsdp_workers.py#L1122) |
| `CriticWorker.compute_values` | `def(self, data)` | [`verl/workers/megatron_workers.py`](../../verl/workers/megatron_workers.py#L683) |
| `CriticWorker.init_model` | `def(self)` | [`verl/workers/fsdp_workers.py`](../../verl/workers/fsdp_workers.py#L1095) |
| `CriticWorker.init_model` | `def(self)` | [`verl/workers/megatron_workers.py`](../../verl/workers/megatron_workers.py#L631) |
| `CriticWorker.load_checkpoint` | `def(self, local_path, hdfs_path, del_local_after_load)` | [`verl/workers/fsdp_workers.py`](../../verl/workers/fsdp_workers.py#L1194) |
| `CriticWorker.load_checkpoint` | `def(self, checkpoint_path, hdfs_path, del_local_after_load)` | [`verl/workers/megatron_workers.py`](../../verl/workers/megatron_workers.py#L724) |
| `CriticWorker.save_checkpoint` | `def(self, local_path, hdfs_path, global_step, max_ckpt_to_keep)` | [`verl/workers/fsdp_workers.py`](../../verl/workers/fsdp_workers.py#L1181) |
| `CriticWorker.save_checkpoint` | `def(self, checkpoint_path, hdfs_path, global_steps, max_ckpt_to_keep)` | [`verl/workers/megatron_workers.py`](../../verl/workers/megatron_workers.py#L734) |
| `CriticWorker.update_critic` | `def(self, data)` | [`verl/workers/fsdp_workers.py`](../../verl/workers/fsdp_workers.py#L1145) |
| `CriticWorker.update_critic` | `def(self, data)` | [`verl/workers/megatron_workers.py`](../../verl/workers/megatron_workers.py#L699) |
| `CSVOutputFormat.__init__` | `def(self, filename)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L122) |
| `CSVOutputFormat.close` | `def(self)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L153) |
| `CSVOutputFormat.writekvs` | `def(self, kvs)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L127) |
| `custom_compare_` | `def(output, ground_truth)` | [`verl/utils/reward_score/prime_code/testing_util.py`](../../verl/utils/reward_score/prime_code/testing_util.py#L509) |
| `DAPORewardManager.__call__` | `def(self, data, return_dict)` | [`verl/workers/reward_manager/dapo.py`](../../verl/workers/reward_manager/dapo.py#L45) |
| `DAPORewardManager.__init__` | `def(self, tokenizer, num_examine, compute_score, reward_fn_key, max_resp_len, overlong_buffer_cfg)` | [`verl/workers/reward_manager/dapo.py`](../../verl/workers/reward_manager/dapo.py#L26) |
| `data_collator` | `def(batch)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/train_choice_il.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/train_choice_il.py#L189) |
| `data_collator` | `def(batch)` | [`agent_system/environments/env_package/webshop/webshop/transfer/app.py`](../../agent_system/environments/env_package/webshop/webshop/transfer/app.py#L43) |
| `DataParallelPPOActor.__init__` | `def(self, config, actor_module, actor_optimizer)` | [`verl/workers/actor/dp_actor.py`](../../verl/workers/actor/dp_actor.py#L56) |
| `DataParallelPPOActor._forward_micro_batch` | `def(self, micro_batch, temperature, calculate_entropy, topk_k, topk_ids)` | [`verl/workers/actor/dp_actor.py`](../../verl/workers/actor/dp_actor.py#L77) |
| `DataParallelPPOActor._optimizer_step` | `def(self)` | [`verl/workers/actor/dp_actor.py`](../../verl/workers/actor/dp_actor.py#L279) |
| `DataParallelPPOActor.compute_log_prob` | `def(self, data, calculate_entropy)` | [`verl/workers/actor/dp_actor.py`](../../verl/workers/actor/dp_actor.py#L298) |
| `DataParallelPPOActor.compute_topk_log_prob` | `def(self, data, topk_k)` | [`verl/workers/actor/dp_actor.py`](../../verl/workers/actor/dp_actor.py#L362) |
| `DataParallelPPOActor.update_policy` | `def(self, data)` | [`verl/workers/actor/dp_actor.py`](../../verl/workers/actor/dp_actor.py#L410) |
| `DataParallelPPOCritic.__init__` | `def(self, config, critic_module, critic_optimizer)` | [`verl/workers/critic/dp_critic.py`](../../verl/workers/critic/dp_critic.py#L51) |
| `DataParallelPPOCritic._forward_micro_batch` | `def(self, micro_batch)` | [`verl/workers/critic/dp_critic.py`](../../verl/workers/critic/dp_critic.py#L61) |
| `DataParallelPPOCritic._optimizer_step` | `def(self)` | [`verl/workers/critic/dp_critic.py`](../../verl/workers/critic/dp_critic.py#L120) |
| `DataParallelPPOCritic.compute_values` | `def(self, data)` | [`verl/workers/critic/dp_critic.py`](../../verl/workers/critic/dp_critic.py#L139) |
| `DataParallelPPOCritic.update_critic` | `def(self, data)` | [`verl/workers/critic/dp_critic.py`](../../verl/workers/critic/dp_critic.py#L180) |
| `DataParallelPRIMERewardModel.__init__` | `def(self, config, reward_module, ref_module, reward_optimizer)` | [`recipe/prime/prime_dp_rm.py`](../../recipe/prime/prime_dp_rm.py#L38) |
| `DataParallelPRIMERewardModel._forward_micro_batch` | `def(self, micro_batch, prompt_length)` | [`recipe/prime/prime_dp_rm.py`](../../recipe/prime/prime_dp_rm.py#L50) |
| `DataParallelPRIMERewardModel._optimizer_step` | `def(self)` | [`recipe/prime/prime_dp_rm.py`](../../recipe/prime/prime_dp_rm.py#L199) |
| `DataParallelPRIMERewardModel.compute_rm_score` | `def(self, data)` | [`recipe/prime/prime_dp_rm.py`](../../recipe/prime/prime_dp_rm.py#L215) |
| `DataParallelPRIMERewardModel.prime_norm` | `def(self, token_level_scores)` | [`recipe/prime/prime_dp_rm.py`](../../recipe/prime/prime_dp_rm.py#L209) |
| `DataParallelPRIMERewardModel.update_rm` | `def(self, data)` | [`recipe/prime/prime_dp_rm.py`](../../recipe/prime/prime_dp_rm.py#L258) |
| `DataParallelSPPOActor.update_policy` | `def(self, data)` | [`recipe/sppo/dp_actor.py`](../../recipe/sppo/dp_actor.py#L61) |
| `DataProto.__getitem__` | `def(self, item)` | [`verl/protocol.py`](../../verl/protocol.py#L226) |
| `DataProto.__getstate__` | `def(self)` | [`verl/protocol.py`](../../verl/protocol.py#L260) |
| `DataProto.__len__` | `def(self)` | [`verl/protocol.py`](../../verl/protocol.py#L217) |
| `DataProto.__post_init__` | `def(self)` | [`verl/protocol.py`](../../verl/protocol.py#L213) |
| `DataProto.__setstate__` | `def(self, data)` | [`verl/protocol.py`](../../verl/protocol.py#L271) |
| `DataProto.check_consistency` | `def(self)` | [`verl/protocol.py`](../../verl/protocol.py#L309) |
| `DataProto.chunk` | `def(self, chunks)` | [`verl/protocol.py`](../../verl/protocol.py#L652) |
| `DataProto.concat` | `def(data)` | [`verl/protocol.py`](../../verl/protocol.py#L690) |
| `DataProto.from_dict` | `def(cls, tensors, non_tensors, meta_info, num_batch_dims, auto_padding)` | [`verl/protocol.py`](../../verl/protocol.py#L346) |
| `DataProto.from_single_dict` | `def(cls, data, meta_info, auto_padding)` | [`verl/protocol.py`](../../verl/protocol.py#L330) |
| `DataProto.is_padding_enabled` | `def(self)` | [`verl/protocol.py`](../../verl/protocol.py#L628) |
| `DataProto.load_from_disk` | `def(filepath)` | [`verl/protocol.py`](../../verl/protocol.py#L286) |
| `DataProto.make_iterator` | `def(self, mini_batch_size, epochs, seed, dataloader_kwargs)` | [`verl/protocol.py`](../../verl/protocol.py#L593) |
| `DataProto.make_iterator.get_data` | `def()` | [`verl/protocol.py`](../../verl/protocol.py#L620) |
| `DataProto.padding` | `def(self, padding_size, padding_candidate)` | [`verl/protocol.py`](../../verl/protocol.py#L637) |
| `DataProto.pop` | `def(self, batch_keys, non_tensor_batch_keys, meta_info_keys)` | [`verl/protocol.py`](../../verl/protocol.py#L516) |
| `DataProto.print_size` | `def(self, prefix)` | [`verl/protocol.py`](../../verl/protocol.py#L291) |
| `DataProto.rename` | `def(self, old_keys, new_keys)` | [`verl/protocol.py`](../../verl/protocol.py#L549) |
| `DataProto.rename.validate_input` | `def(keys)` | [`verl/protocol.py`](../../verl/protocol.py#L554) |
| `DataProto.reorder` | `def(self, indices)` | [`verl/protocol.py`](../../verl/protocol.py#L712) |
| `DataProto.repeat` | `def(self, repeat_times, interleave)` | [`verl/protocol.py`](../../verl/protocol.py#L720) |
| `DataProto.sample_level_repeat` | `def(self, repeat_times)` | [`verl/protocol.py`](../../verl/protocol.py#L796) |
| `DataProto.save_to_disk` | `def(self, filepath)` | [`verl/protocol.py`](../../verl/protocol.py#L281) |
| `DataProto.select` | `def(self, batch_keys, non_tensor_batch_keys, meta_info_keys, deepcopy)` | [`verl/protocol.py`](../../verl/protocol.py#L399) |
| `DataProto.select_idxs` | `def(self, idxs)` | [`verl/protocol.py`](../../verl/protocol.py#L434) |
| `DataProto.slice` | `def(self, start, end, step)` | [`verl/protocol.py`](../../verl/protocol.py#L470) |
| `DataProto.to` | `def(self, device)` | [`verl/protocol.py`](../../verl/protocol.py#L385) |
| `DataProto.unfold_column_chunks` | `def(self, n_split, split_keys)` | [`verl/protocol.py`](../../verl/protocol.py#L759) |
| `DataProto.union` | `def(self, other)` | [`verl/protocol.py`](../../verl/protocol.py#L574) |
| `DataProtoFuture.chunk` | `def(self, chunks)` | [`verl/protocol.py`](../../verl/protocol.py#L864) |
| `DataProtoFuture.chunk.dispatch_fn` | `def(x, i, chunks)` | [`verl/protocol.py`](../../verl/protocol.py#L870) |
| `DataProtoFuture.concat` | `def(data)` | [`verl/protocol.py`](../../verl/protocol.py#L860) |
| `DataProtoFuture.get` | `def(self)` | [`verl/protocol.py`](../../verl/protocol.py#L877) |
| `debug` | `def(*args)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L262) |
| `DecoratorLoggerBase.__init__` | `def(self, role, logger, level, rank, log_only_rank_0)` | [`verl/utils/logger/aggregate_logger.py`](../../verl/utils/logger/aggregate_logger.py#L47) |
| `DecoratorLoggerBase.log_by_logging` | `def(self, log_str)` | [`verl/utils/logger/aggregate_logger.py`](../../verl/utils/logger/aggregate_logger.py#L61) |
| `DecoratorLoggerBase.log_by_print` | `def(self, log_str)` | [`verl/utils/logger/aggregate_logger.py`](../../verl/utils/logger/aggregate_logger.py#L57) |
| `DecoratorTestWorker.__init__` | `def(self, initial_value)` | [`tests/ray_cpu/test_decorator.py`](../../tests/ray_cpu/test_decorator.py#L40) |
| `DecoratorTestWorker.async_dp_compute` | `async def(self, data)` | [`tests/ray_cpu/test_decorator.py`](../../tests/ray_cpu/test_decorator.py#L56) |
| `DecoratorTestWorker.dp_compute` | `def(self, data)` | [`tests/ray_cpu/test_decorator.py`](../../tests/ray_cpu/test_decorator.py#L48) |
| `DeepseekV3Model.get_rope_scaling_args` | `def(self)` | [`verl/models/mcore/model_initializer.py`](../../verl/models/mcore/model_initializer.py#L177) |
| `DeepseekV3Model.get_transformer_layer_spec` | `def(self, vp_stage)` | [`verl/models/mcore/model_initializer.py`](../../verl/models/mcore/model_initializer.py#L172) |
| `DeepseekV3Model.initialize` | `def(self, **kwargs)` | [`verl/models/mcore/model_initializer.py`](../../verl/models/mcore/model_initializer.py#L182) |
| `default_compute_score` | `def(data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore)` | [`verl/utils/reward_score/__init__.py`](../../verl/utils/reward_score/__init__.py#L19) |
| `default_tp_concat_fn` | `def(layer_name_mapping, name, train_params, infer_params, model_config, convert_qkv_gate_up_by_simple_split)` | [`verl/utils/megatron_utils.py`](../../verl/utils/megatron_utils.py#L669) |
| `DenseModel.get_transformer_layer_spec` | `def(self, vp_stage)` | [`verl/models/mcore/model_initializer.py`](../../verl/models/mcore/model_initializer.py#L102) |
| `DenseRetriever.__init__` | `def(self, config)` | [`examples/search/retriever/retrieval_server.py`](../../examples/search/retriever/retrieval_server.py#L196) |
| `DenseRetriever._batch_search` | `def(self, query_list, num, return_score)` | [`examples/search/retriever/retrieval_server.py`](../../examples/search/retriever/retrieval_server.py#L229) |
| `DenseRetriever._search` | `def(self, query, num, return_score)` | [`examples/search/retriever/retrieval_server.py`](../../examples/search/retriever/retrieval_server.py#L216) |
| `deprecated` | `def(replacement)` | [`verl/utils/import_utils.py`](../../verl/utils/import_utils.py#L124) |
| `deprecated.decorator` | `def(obj)` | [`verl/utils/import_utils.py`](../../verl/utils/import_utils.py#L127) |
| `deprecated.decorator.wrapped` | `def(*args, **kwargs)` | [`verl/utils/import_utils.py`](../../verl/utils/import_utils.py#L147) |
| `deprecated.decorator.wrapped_init` | `def(self, *args, **kwargs)` | [`verl/utils/import_utils.py`](../../verl/utils/import_utils.py#L134) |
| `depth_first_search` | `def(room_state, room_structure, box_mapping, box_swaps, last_pull, ttl, action_sequence)` | [`agent_system/environments/env_package/sokoban/sokoban/room_utils.py`](../../agent_system/environments/env_package/sokoban/sokoban/room_utils.py#L446) |
| `dict_to_fake_html` | `def(data, page_type, asin, sub_page_type, options, prod_map, query)` | [`agent_system/environments/env_package/webshop/webshop/transfer/webshop_lite.py`](../../agent_system/environments/env_package/webshop/webshop/transfer/webshop_lite.py#L89) |
| `DigitCompletion.__init__` | `def(self, max_number, max_diff, max_num_in_response, seed)` | [`tests/e2e/envs/digit_completion/task.py`](../../tests/e2e/envs/digit_completion/task.py#L35) |
| `DigitCompletion.__str__` | `def(self)` | [`tests/e2e/envs/digit_completion/task.py`](../../tests/e2e/envs/digit_completion/task.py#L56) |
| `DigitCompletion.add` | `def(self, a, b)` | [`tests/e2e/envs/digit_completion/task.py`](../../tests/e2e/envs/digit_completion/task.py#L76) |
| `DigitCompletion.get_all_prompts` | `def(self)` | [`tests/e2e/envs/digit_completion/task.py`](../../tests/e2e/envs/digit_completion/task.py#L79) |
| `DigitCompletion.get_state` | `def(self)` | [`tests/e2e/envs/digit_completion/task.py`](../../tests/e2e/envs/digit_completion/task.py#L59) |
| `DigitCompletion.prompt_length` | `def(self)` | [`tests/e2e/envs/digit_completion/task.py`](../../tests/e2e/envs/digit_completion/task.py#L67) |
| `DigitCompletion.response_length` | `def(self)` | [`tests/e2e/envs/digit_completion/task.py`](../../tests/e2e/envs/digit_completion/task.py#L71) |
| `DigitCompletion.sample_batch_str_prompts` | `def(self, batch_size)` | [`tests/e2e/envs/digit_completion/task.py`](../../tests/e2e/envs/digit_completion/task.py#L98) |
| `DigitCompletion.sample_str_prompts` | `def(self)` | [`tests/e2e/envs/digit_completion/task.py`](../../tests/e2e/envs/digit_completion/task.py#L89) |
| `DigitCompletion.set_state` | `def(self, state)` | [`tests/e2e/envs/digit_completion/task.py`](../../tests/e2e/envs/digit_completion/task.py#L62) |
| `discount_reward` | `def(transitions, last_values, gamma)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/agent.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/agent.py#L18) |
| `dispatch_all_to_all` | `def(worker_group, *args, **kwargs)` | [`verl/single_controller/base/decorator.py`](../../verl/single_controller/base/decorator.py#L142) |
| `dispatch_dp_compute` | `def(worker_group, *args, **kwargs)` | [`verl/single_controller/base/decorator.py`](../../verl/single_controller/base/decorator.py#L344) |
| `dispatch_dp_compute_data_proto` | `def(worker_group, *args, **kwargs)` | [`verl/single_controller/base/decorator.py`](../../verl/single_controller/base/decorator.py#L363) |
| `dispatch_dp_compute_data_proto_with_func` | `def(worker_group, *args, **kwargs)` | [`verl/single_controller/base/decorator.py`](../../verl/single_controller/base/decorator.py#L376) |
| `dispatch_megatron_compute` | `def(worker_group, *args, **kwargs)` | [`verl/single_controller/base/decorator.py`](../../verl/single_controller/base/decorator.py#L150) |
| `dispatch_megatron_compute_data_proto` | `def(worker_group, *args, **kwargs)` | [`verl/single_controller/base/decorator.py`](../../verl/single_controller/base/decorator.py#L195) |
| `dispatch_megatron_pp_as_dp` | `def(worker_group, *args, **kwargs)` | [`verl/single_controller/base/decorator.py`](../../verl/single_controller/base/decorator.py#L241) |
| `dispatch_megatron_pp_as_dp_data_proto` | `def(worker_group, *args, **kwargs)` | [`verl/single_controller/base/decorator.py`](../../verl/single_controller/base/decorator.py#L324) |
| `dispatch_one_to_all` | `def(worker_group, *args, **kwargs)` | [`verl/single_controller/base/decorator.py`](../../verl/single_controller/base/decorator.py#L132) |
| `distributed_masked_mean` | `def(local_tensor, local_mask)` | [`verl/utils/torch_functional.py`](../../verl/utils/torch_functional.py#L700) |
| `distributed_mean_max_min_std` | `def(local_tensor, compute_max, compute_min, compute_std)` | [`verl/utils/torch_functional.py`](../../verl/utils/torch_functional.py#L657) |
| `done` | `def(asin, options, session_id, **kwargs)` | [`agent_system/environments/env_package/webshop/webshop/transfer/webshop_lite.py`](../../agent_system/environments/env_package/webshop/webshop/transfer/webshop_lite.py#L69) |
| `done` | `def(session_id, asin, options)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/app.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/app.py#L220) |
| `download_files_distributed` | `def(download_fn)` | [`verl/utils/dataset/rm_dataset.py`](../../verl/utils/dataset/rm_dataset.py#L25) |
| `draw_card` | `def(np_random)` | [`agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/blackjack.py`](../../agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/blackjack.py#L46) |
| `draw_card_with_info` | `def(np_random)` | [`agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/blackjack.py`](../../agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/blackjack.py#L21) |
| `draw_hand` | `def(np_random)` | [`agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/blackjack.py`](../../agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/blackjack.py#L51) |
| `draw_hand_with_info` | `def(np_random)` | [`agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/blackjack.py`](../../agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/blackjack.py#L32) |
| `dummy_direct_rollout_call` | `def(worker_group, *args, **kwargs)` | [`verl/single_controller/base/decorator.py`](../../verl/single_controller/base/decorator.py#L138) |
| `DummyWorker.__init__` | `def(self)` | [`tests/ray_gpu/test_data_transfer.py`](../../tests/ray_gpu/test_data_transfer.py#L33) |
| `DummyWorker.do_nothing` | `def(self, data)` | [`tests/ray_gpu/test_data_transfer.py`](../../tests/ray_gpu/test_data_transfer.py#L38) |
| `dump_data` | `def(data, name)` | [`verl/utils/debug/trajectory_tracker.py`](../../verl/utils/debug/trajectory_tracker.py#L69) |
| `dumpkvs` | `def()` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L241) |
| `duplicate` | `def(output, mask, lens, act_sizes)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/models/modules.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/models/modules.py#L8) |
| `DynamicEnum.__init__` | `def(self, name, value)` | [`verl/utils/py_functional.py`](../../verl/utils/py_functional.py#L231) |
| `DynamicEnum.__reduce_ex__` | `def(self, protocol)` | [`verl/utils/py_functional.py`](../../verl/utils/py_functional.py#L238) |
| `DynamicEnum.__repr__` | `def(self)` | [`verl/utils/py_functional.py`](../../verl/utils/py_functional.py#L235) |
| `DynamicEnum.from_name` | `def(cls, name)` | [`verl/utils/py_functional.py`](../../verl/utils/py_functional.py#L266) |
| `DynamicEnum.register` | `def(cls, name)` | [`verl/utils/py_functional.py`](../../verl/utils/py_functional.py#L248) |
| `DynamicEnum.remove` | `def(cls, name)` | [`verl/utils/py_functional.py`](../../verl/utils/py_functional.py#L259) |
| `DynamicEnumMeta.__contains__` | `def(cls, item)` | [`verl/utils/py_functional.py`](../../verl/utils/py_functional.py#L207) |
| `DynamicEnumMeta.__getitem__` | `def(cls, name)` | [`verl/utils/py_functional.py`](../../verl/utils/py_functional.py#L213) |
| `DynamicEnumMeta.__iter__` | `def(cls)` | [`verl/utils/py_functional.py`](../../verl/utils/py_functional.py#L204) |
| `DynamicEnumMeta.__reduce_ex__` | `def(cls, protocol)` | [`verl/utils/py_functional.py`](../../verl/utils/py_functional.py#L216) |
| `DynamicEnumMeta.names` | `def(cls)` | [`verl/utils/py_functional.py`](../../verl/utils/py_functional.py#L220) |
| `DynamicEnumMeta.values` | `def(cls)` | [`verl/utils/py_functional.py`](../../verl/utils/py_functional.py#L223) |
| `em_check` | `def(prediction, golden_answers)` | [`verl/utils/reward_score/search_r1_like_qa_em.py`](../../verl/utils/reward_score/search_r1_like_qa_em.py#L40) |
| `enable_activation_offloading` | `def(model, strategy, enable_ckpt)` | [`verl/utils/activation_offload.py`](../../verl/utils/activation_offload.py#L494) |
| `enable_activation_offloading.get_layers` | `def(module)` | [`verl/utils/activation_offload.py`](../../verl/utils/activation_offload.py#L518) |
| `enabled` | `def()` | [`verl/utils/gpu_profiler.py`](../../verl/utils/gpu_profiler.py#L71) |
| `Encoder.__init__` | `def(self, model_name, model_path, pooling_method, max_length, use_fp16)` | [`examples/search/retriever/retrieval_server.py`](../../examples/search/retriever/retrieval_server.py#L59) |
| `Encoder.encode` | `def(self, query_list, is_query)` | [`examples/search/retriever/retrieval_server.py`](../../examples/search/retriever/retrieval_server.py#L70) |
| `EncoderRNN.__init__` | `def(self, input_size, num_units, nlayers, concat, bidir, layernorm, return_last)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/models/modules.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/models/modules.py#L32) |
| `EncoderRNN.forward` | `def(self, inputs, input_lengths)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/models/modules.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/models/modules.py#L78) |
| `EncoderRNN.get_init` | `def(self, bsz, i)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/models/modules.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/models/modules.py#L75) |
| `EncoderRNN.reset_parameters` | `def(self)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/models/modules.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/models/modules.py#L62) |
| `enforce_expected_config` | `def(config, expect_file, tag)` | [`verl/utils/expected_config.py`](../../verl/utils/expected_config.py#L108) |
| `entropy_from_logits` | `def(logits)` | [`verl/utils/torch_functional.py`](../../verl/utils/torch_functional.py#L124) |
| `EnvironmentManagerBase.__init__` | `def(self, envs, projection_f, config)` | [`agent_system/environments/base.py`](../../agent_system/environments/base.py#L35) |
| `EnvironmentManagerBase._process_batch` | `def(self, batch_idx, total_batch_list, total_infos, success)` | [`agent_system/environments/base.py`](../../agent_system/environments/base.py#L135) |
| `EnvironmentManagerBase.build_text_obs` | `def(self)` | [`agent_system/environments/base.py`](../../agent_system/environments/base.py#L99) |
| `EnvironmentManagerBase.close` | `def(self)` | [`agent_system/environments/base.py`](../../agent_system/environments/base.py#L108) |
| `EnvironmentManagerBase.reset` | `def(self, kwargs)` | [`agent_system/environments/base.py`](../../agent_system/environments/base.py#L48) |
| `EnvironmentManagerBase.save_image` | `def(self, image, step)` | [`agent_system/environments/base.py`](../../agent_system/environments/base.py#L144) |
| `EnvironmentManagerBase.step` | `def(self, text_actions)` | [`agent_system/environments/base.py`](../../agent_system/environments/base.py#L63) |
| `EnvironmentManagerBase.success_evaluator` | `def(self, *args, **kwargs)` | [`agent_system/environments/base.py`](../../agent_system/environments/base.py#L114) |
| `episode` | `def(model, idx, verbose, softmax, rule, bart_model)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/test.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/test.py#L71) |
| `episode_norm_reward` | `def(token_level_rewards, response_mask, index, traj_index, epsilon, remove_std, compute_mean_std_cross_steps)` | [`gigpo/core_gigpo.py`](../../gigpo/core_gigpo.py#L174) |
| `EpisodeRewardManager.__call__` | `def(self, data, return_dict)` | [`agent_system/reward_manager/episode.py`](../../agent_system/reward_manager/episode.py#L29) |
| `EpisodeRewardManager.__init__` | `def(self, tokenizer, num_examine, normalize_by_length)` | [`agent_system/reward_manager/episode.py`](../../agent_system/reward_manager/episode.py#L24) |
| `error` | `def(*args)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L274) |
| `evaluate` | `def(agent, env, split, nb_episodes)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/train_rl.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/train_rl.py#L24) |
| `evaluate_episode` | `def(agent, env, split, method, idx)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/train_rl.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/train_rl.py#L40) |
| `example_map_fn` | `def(example, idx, process_fn, data_source, ability, split)` | [`recipe/r1/data_process.py`](../../recipe/r1/data_process.py#L27) |
| `ExecutionWorker.__init__` | `def(self, enable_global_rate_limit, rate_limit)` | [`verl/tools/sandbox_fusion_tools.py`](../../verl/tools/sandbox_fusion_tools.py#L66) |
| `ExecutionWorker._init_rate_limit` | `def(self, rate_limit)` | [`verl/tools/sandbox_fusion_tools.py`](../../verl/tools/sandbox_fusion_tools.py#L69) |
| `ExecutionWorker.execute` | `def(self, fn, *fn_args, **fn_kwargs)` | [`verl/tools/sandbox_fusion_tools.py`](../../verl/tools/sandbox_fusion_tools.py#L77) |
| `ExecutionWorker.ping` | `def(self)` | [`verl/tools/sandbox_fusion_tools.py`](../../verl/tools/sandbox_fusion_tools.py#L74) |
| `exists` | `def(path, **kwargs)` | [`verl/utils/hdfs_io.py`](../../verl/utils/hdfs_io.py#L27) |
| `ExternalRayDistributedExecutor._init_executor` | `def(self)` | [`verl/workers/rollout/vllm_rollout/vllm_async_server.py`](../../verl/workers/rollout/vllm_rollout/vllm_async_server.py#L44) |
| `ExternalRayDistributedExecutor._init_executor.get_pg_index_and_local_rank` | `def(actor_name)` | [`verl/workers/rollout/vllm_rollout/vllm_async_server.py`](../../verl/workers/rollout/vllm_rollout/vllm_async_server.py#L59) |
| `ExternalRayDistributedExecutor.check_health` | `def(self)` | [`verl/workers/rollout/vllm_rollout/vllm_async_server.py`](../../verl/workers/rollout/vllm_rollout/vllm_async_server.py#L101) |
| `ExternalRayDistributedExecutor.collective_rpc` | `def(self, method, timeout, args, kwargs)` | [`verl/workers/rollout/vllm_rollout/vllm_async_server.py`](../../verl/workers/rollout/vllm_rollout/vllm_async_server.py#L83) |
| `extract_pg_from_exist` | `def(resource_pools, src_role_names, resource_pool)` | [`verl/single_controller/ray/base.py`](../../verl/single_controller/ray/base.py#L131) |
| `extract_reward_from_line` | `def(line)` | [`tests/e2e/check_results.py`](../../tests/e2e/check_results.py#L20) |
| `extract_solution` | `def(solution_str)` | [`examples/data_preprocess/gsm8k.py`](../../examples/data_preprocess/gsm8k.py#L27) |
| `extract_solution` | `def(solution_str)` | [`examples/data_preprocess/gsm8k_multiturn_w_tool.py`](../../examples/data_preprocess/gsm8k_multiturn_w_tool.py#L29) |
| `extract_solution` | `def(solution_str)` | [`examples/data_preprocess/math_dataset.py`](../../examples/data_preprocess/math_dataset.py#L27) |
| `extract_solution` | `def(solution_str, method)` | [`verl/utils/reward_score/gsm8k.py`](../../verl/utils/reward_score/gsm8k.py#L18) |
| `extract_solution` | `def(solution_str)` | [`verl/utils/reward_score/search_r1_like_qa_em.py`](../../verl/utils/reward_score/search_r1_like_qa_em.py#L66) |
| `extract_step` | `def(path)` | [`verl/trainer/fsdp_sft_trainer.py`](../../verl/trainer/fsdp_sft_trainer.py#L80) |
| `EZPointEnv.__init__` | `def(self, target_points)` | [`agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/ezpoints.py`](../../agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/ezpoints.py#L51) |
| `EZPointEnv._card_num_to_str` | `def(self, num)` | [`agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/ezpoints.py`](../../agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/ezpoints.py#L127) |
| `EZPointEnv._evaluate_formula` | `def(self)` | [`agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/ezpoints.py`](../../agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/ezpoints.py#L143) |
| `EZPointEnv._generate_cards` | `def(self)` | [`agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/ezpoints.py`](../../agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/ezpoints.py#L104) |
| `EZPointEnv._get_observation` | `def(self)` | [`agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/ezpoints.py`](../../agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/ezpoints.py#L161) |
| `EZPointEnv._is_valid_action` | `def(self, action)` | [`agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/ezpoints.py`](../../agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/ezpoints.py#L133) |
| `EZPointEnv._terminate_step` | `def(self, reward, info_key, is_truncated)` | [`agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/ezpoints.py`](../../agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/ezpoints.py#L158) |
| `EZPointEnv.reset` | `def(self, seed, options)` | [`agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/ezpoints.py`](../../agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/ezpoints.py#L62) |
| `EZPointEnv.set_action_space` | `def(self)` | [`agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/ezpoints.py`](../../agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/ezpoints.py#L57) |
| `EZPointEnv.step` | `def(self, action)` | [`agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/ezpoints.py`](../../agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/ezpoints.py#L81) |
| `FakeEnvs.__init__` | `def(self)` | [`tests/ray_cpu/test_rollout_speedup_mechanisms.py`](../../tests/ray_cpu/test_rollout_speedup_mechanisms.py#L231) |
| `FakeEnvs.reset` | `def(self, kwargs)` | [`tests/ray_cpu/test_rollout_speedup_mechanisms.py`](../../tests/ray_cpu/test_rollout_speedup_mechanisms.py#L234) |
| `FakeWorkerGroup.__init__` | `def(self, world_size, response_length)` | [`tests/ray_cpu/test_rollout_speedup_mechanisms.py`](../../tests/ray_cpu/test_rollout_speedup_mechanisms.py#L63) |
| `FakeWorkerGroup.compute_log_prob` | `def(self, data)` | [`tests/ray_cpu/test_rollout_speedup_mechanisms.py`](../../tests/ray_cpu/test_rollout_speedup_mechanisms.py#L68) |
| `fast_forward_env_schedules` | `def(envs, num_resets)` | [`agent_system/environments/resume.py`](../../agent_system/environments/resume.py#L64) |
| `filter_group_data` | `def(batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings, config, last_try)` | [`agent_system/multi_turn_rollout/utils.py`](../../agent_system/multi_turn_rollout/utils.py#L204) |
| `find_latest_ckpt_path` | `def(path, directory_format)` | [`verl/utils/checkpoint/checkpoint_manager.py`](../../verl/utils/checkpoint/checkpoint_manager.py#L135) |
| `FinishReasonTypeEnum.from_str` | `def(cls, value)` | [`verl/workers/rollout/schemas.py`](../../verl/workers/rollout/schemas.py#L35) |
| `FIREvLLMRollout.__init__` | `def(self, actor_module, config, tokenizer, model_hf_config, **kwargs)` | [`verl/workers/rollout/vllm_rollout/fire_vllm_rollout.py`](../../verl/workers/rollout/vllm_rollout/fire_vllm_rollout.py#L58) |
| `FIREvLLMRollout.generate_sequences` | `def(self, prompts, **kwargs)` | [`verl/workers/rollout/vllm_rollout/fire_vllm_rollout.py`](../../verl/workers/rollout/vllm_rollout/fire_vllm_rollout.py#L111) |
| `FIREvLLMRollout.update_sampling_params` | `def(self, **kwargs)` | [`verl/workers/rollout/vllm_rollout/fire_vllm_rollout.py`](../../verl/workers/rollout/vllm_rollout/fire_vllm_rollout.py#L84) |
| `fit` | `def(self)` | [`examples/split_placement/split_monkey_patch.py`](../../examples/split_placement/split_monkey_patch.py#L37) |
| `fix_a_slash_b` | `def(string)` | [`verl/utils/reward_score/math.py`](../../verl/utils/reward_score/math.py#L122) |
| `fix_fracs` | `def(string)` | [`verl/utils/reward_score/math.py`](../../verl/utils/reward_score/math.py#L90) |
| `fix_sqrt` | `def(string)` | [`verl/utils/reward_score/math.py`](../../verl/utils/reward_score/math.py#L147) |
| `FixedKLController.__init__` | `def(self, kl_coef)` | [`recipe/spin/core_algos.py`](../../recipe/spin/core_algos.py#L43) |
| `FixedKLController.__init__` | `def(self, kl_coef)` | [`verl/trainer/ppo/core_algos.py`](../../verl/trainer/ppo/core_algos.py#L50) |
| `FixedKLController.update` | `def(self, current_kl, n_steps)` | [`recipe/spin/core_algos.py`](../../recipe/spin/core_algos.py#L46) |
| `FixedKLController.update` | `def(self, current_kl, n_steps)` | [`verl/trainer/ppo/core_algos.py`](../../verl/trainer/ppo/core_algos.py#L53) |
| `flash_attn_supports_top_left_mask` | `def()` | [`verl/utils/transformers_compat.py`](../../verl/utils/transformers_compat.py#L32) |
| `FlopsCounter.__init__` | `def(self, config)` | [`verl/utils/flops_counter.py`](../../verl/utils/flops_counter.py#L118) |
| `FlopsCounter._estimate_apertus_flops` | `def(self, tokens_sum, batch_seqlens, delta_time)` | [`verl/utils/flops_counter.py`](../../verl/utils/flops_counter.py#L340) |
| `FlopsCounter._estimate_deepseek_v3_flops` | `def(self, tokens_sum, batch_seqlens, delta_time)` | [`verl/utils/flops_counter.py`](../../verl/utils/flops_counter.py#L183) |
| `FlopsCounter._estimate_gemma3_flops` | `def(self, tokens_sum, batch_seqlens, delta_time)` | [`verl/utils/flops_counter.py`](../../verl/utils/flops_counter.py#L273) |
| `FlopsCounter._estimate_qwen2_flops` | `def(self, tokens_sum, batch_seqlens, delta_time)` | [`verl/utils/flops_counter.py`](../../verl/utils/flops_counter.py#L149) |
| `FlopsCounter._estimate_qwen2_moe_flops` | `def(self, tokens_sum, batch_seqlens, delta_time)` | [`verl/utils/flops_counter.py`](../../verl/utils/flops_counter.py#L237) |
| `FlopsCounter._estimate_unknown_flops` | `def(self, tokens_sum, batch_seqlens, delta_time)` | [`verl/utils/flops_counter.py`](../../verl/utils/flops_counter.py#L146) |
| `FlopsCounter.estimate_flops` | `def(self, batch_seqlens, delta_time)` | [`verl/utils/flops_counter.py`](../../verl/utils/flops_counter.py#L379) |
| `fold_batch_dim` | `def(data, new_batch_size)` | [`verl/protocol.py`](../../verl/protocol.py#L140) |
| `format_intervals` | `def(prediction)` | [`verl/utils/reward_score/prime_math/grader.py`](../../verl/utils/reward_score/prime_math/grader.py#L319) |
| `format_reward` | `def(predict_str)` | [`verl/utils/reward_score/geo3k.py`](../../verl/utils/reward_score/geo3k.py#L20) |
| `forward_for_ppo` | `def(self, input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict, cache_position, logits_to_keep, temperature, **loss_kwargs)` | [`verl/models/transformers/llama.py`](../../verl/models/transformers/llama.py#L241) |
| `forward_for_ppo` | `def(self, input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw, rope_deltas, cache_position, second_per_grid_ts, temperature, **loss_kwargs)` | [`verl/models/transformers/qwen2_5_vl.py`](../../verl/models/transformers/qwen2_5_vl.py#L31) |
| `forward_with_normal_backend` | `def(self, input_ids, labels, temperature, **kwargs)` | [`verl/models/transformers/qwen2_vl.py`](../../verl/models/transformers/qwen2_vl.py#L466) |
| `forward_with_normal_backend` | `def(self, input_ids, labels, temperature, **kwargs)` | [`verl/models/transformers/qwen3_vl.py`](../../verl/models/transformers/qwen3_vl.py#L254) |
| `forward_with_torch_backend` | `def(self, input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict, cache_position, logits_to_keep, temperature, **kwargs)` | [`verl/models/transformers/dense_common.py`](../../verl/models/transformers/dense_common.py#L51) |
| `forward_with_torch_backend` | `def(self, input_ids, labels, temperature, **kwargs)` | [`verl/models/transformers/qwen2_vl.py`](../../verl/models/transformers/qwen2_vl.py#L483) |
| `forward_with_torch_backend` | `def(self, input_ids, labels, temperature, **kwargs)` | [`verl/models/transformers/qwen3_vl.py`](../../verl/models/transformers/qwen3_vl.py#L271) |
| `forward_with_triton_backend` | `def(self, input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict, cache_position, logits_to_keep, temperature, **kwargs)` | [`verl/models/transformers/dense_common.py`](../../verl/models/transformers/dense_common.py#L100) |
| `forward_with_triton_backend` | `def(self, input_ids, labels, temperature, **kwargs)` | [`verl/models/transformers/qwen2_vl.py`](../../verl/models/transformers/qwen2_vl.py#L517) |
| `forward_with_triton_backend` | `def(self, input_ids, labels, temperature, **kwargs)` | [`verl/models/transformers/qwen3_vl.py`](../../verl/models/transformers/qwen3_vl.py#L305) |
| `fsdp2_clip_grad_norm_` | `def(parameters, max_norm, norm_type, error_if_nonfinite, foreach)` | [`verl/utils/fsdp_utils.py`](../../verl/utils/fsdp_utils.py#L447) |
| `fsdp2_load_full_state_dict` | `def(model, full_state, device_mesh, cpu_offload)` | [`verl/utils/fsdp_utils.py`](../../verl/utils/fsdp_utils.py#L394) |
| `fsdp_version` | `def(model)` | [`verl/utils/fsdp_utils.py`](../../verl/utils/fsdp_utils.py#L378) |
| `FSDPCheckpointManager.__init__` | `def(self, model, optimizer, lr_scheduler, processing_class, checkpoint_contents, **kwargs)` | [`verl/utils/checkpoint/fsdp_checkpoint_manager.py`](../../verl/utils/checkpoint/fsdp_checkpoint_manager.py#L51) |
| `FSDPCheckpointManager.load_checkpoint` | `def(self, local_path, hdfs_path, del_local_after_load)` | [`verl/utils/checkpoint/fsdp_checkpoint_manager.py`](../../verl/utils/checkpoint/fsdp_checkpoint_manager.py#L76) |
| `FSDPCheckpointManager.save_checkpoint` | `def(self, local_path, hdfs_path, global_step, max_ckpt_to_keep)` | [`verl/utils/checkpoint/fsdp_checkpoint_manager.py`](../../verl/utils/checkpoint/fsdp_checkpoint_manager.py#L129) |
| `FSDPModelMerger._calculate_shard_configuration` | `def(self, mesh, mesh_dim_names)` | [`scripts/model_merger.py`](../../scripts/model_merger.py#L196) |
| `FSDPModelMerger._extract_device_mesh_info` | `def(self, state_dict, world_size)` | [`scripts/model_merger.py`](../../scripts/model_merger.py#L176) |
| `FSDPModelMerger._get_world_size` | `def(self)` | [`scripts/model_merger.py`](../../scripts/model_merger.py#L165) |
| `FSDPModelMerger._load_and_merge_state_dicts` | `def(self, world_size, total_shards, mesh_shape, mesh_dim_names)` | [`scripts/model_merger.py`](../../scripts/model_merger.py#L221) |
| `FSDPModelMerger._load_and_merge_state_dicts.process_one_shard` | `def(rank, model_state_dict_lst)` | [`scripts/model_merger.py`](../../scripts/model_merger.py#L224) |
| `FSDPModelMerger._load_rank_zero_state_dict` | `def(self, world_size)` | [`scripts/model_merger.py`](../../scripts/model_merger.py#L173) |
| `FSDPModelMerger._merge_by_placement` | `def(self, tensors, placement)` | [`scripts/model_merger.py`](../../scripts/model_merger.py#L210) |
| `FSDPModelMerger._test_state_dict` | `def(self, state_dict)` | [`scripts/model_merger.py`](../../scripts/model_merger.py#L305) |
| `FSDPModelMerger.merge_and_save` | `def(self)` | [`scripts/model_merger.py`](../../scripts/model_merger.py#L282) |
| `FSDPParameterFilter.__call__` | `def(self, tensor)` | [`verl/utils/activation_offload.py`](../../verl/utils/activation_offload.py#L43) |
| `FSDPParameterFilter.__init__` | `def(self)` | [`verl/utils/activation_offload.py`](../../verl/utils/activation_offload.py#L40) |
| `FSDPParameterFilter.update_model_parameters` | `def(self, model)` | [`verl/utils/activation_offload.py`](../../verl/utils/activation_offload.py#L46) |
| `FSDPSFTTrainer.__init__` | `def(self, config, device_mesh, ulysses_device_mesh, tokenizer, train_dataset, val_dataset)` | [`verl/trainer/fsdp_sft_trainer.py`](../../verl/trainer/fsdp_sft_trainer.py#L88) |
| `FSDPSFTTrainer._build_dataloader` | `def(self, train_dataset, val_dataset)` | [`verl/trainer/fsdp_sft_trainer.py`](../../verl/trainer/fsdp_sft_trainer.py#L127) |
| `FSDPSFTTrainer._build_model_optimizer` | `def(self)` | [`verl/trainer/fsdp_sft_trainer.py`](../../verl/trainer/fsdp_sft_trainer.py#L168) |
| `FSDPSFTTrainer._compute_loss_and_backward` | `def(self, batch, do_backward)` | [`verl/trainer/fsdp_sft_trainer.py`](../../verl/trainer/fsdp_sft_trainer.py#L303) |
| `FSDPSFTTrainer._normalize_config_bsz` | `def(self)` | [`verl/trainer/fsdp_sft_trainer.py`](../../verl/trainer/fsdp_sft_trainer.py#L116) |
| `FSDPSFTTrainer.fit` | `def(self)` | [`verl/trainer/fsdp_sft_trainer.py`](../../verl/trainer/fsdp_sft_trainer.py#L492) |
| `FSDPSFTTrainer.save_checkpoint` | `def(self, step)` | [`verl/trainer/fsdp_sft_trainer.py`](../../verl/trainer/fsdp_sft_trainer.py#L450) |
| `FSDPSFTTrainer.training_step` | `def(self, batch)` | [`verl/trainer/fsdp_sft_trainer.py`](../../verl/trainer/fsdp_sft_trainer.py#L390) |
| `FSDPSFTTrainer.validation_step` | `def(self, batch)` | [`verl/trainer/fsdp_sft_trainer.py`](../../verl/trainer/fsdp_sft_trainer.py#L439) |
| `FSDPSGLangShardingManager.__enter__` | `def(self)` | [`verl/workers/sharding_manager/fsdp_sglang.py`](../../verl/workers/sharding_manager/fsdp_sglang.py#L92) |
| `FSDPSGLangShardingManager.__exit__` | `def(self, exc_type, exc_value, traceback)` | [`verl/workers/sharding_manager/fsdp_sglang.py`](../../verl/workers/sharding_manager/fsdp_sglang.py#L117) |
| `FSDPSGLangShardingManager.__init__` | `def(self, module, inference_engine, model_config, full_params, device_mesh, offload_param)` | [`verl/workers/sharding_manager/fsdp_sglang.py`](../../verl/workers/sharding_manager/fsdp_sglang.py#L51) |
| `FSDPSGLangShardingManager.postprocess_data` | `def(self, data)` | [`verl/workers/sharding_manager/fsdp_sglang.py`](../../verl/workers/sharding_manager/fsdp_sglang.py#L180) |
| `FSDPSGLangShardingManager.preprocess_data` | `def(self, data)` | [`verl/workers/sharding_manager/fsdp_sglang.py`](../../verl/workers/sharding_manager/fsdp_sglang.py#L169) |
| `FSDPSGLangShardingManager.release_memory` | `def(self)` | [`verl/workers/sharding_manager/fsdp_sglang.py`](../../verl/workers/sharding_manager/fsdp_sglang.py#L165) |
| `FSDPSGLangShardingManager.update_weights` | `def(self, params)` | [`verl/workers/sharding_manager/fsdp_sglang.py`](../../verl/workers/sharding_manager/fsdp_sglang.py#L132) |
| `FSDPUlyssesShardingManager.__enter__` | `def(self)` | [`verl/workers/sharding_manager/fsdp_ulysses.py`](../../verl/workers/sharding_manager/fsdp_ulysses.py#L37) |
| `FSDPUlyssesShardingManager.__exit__` | `def(self, exc_type, exc_value, traceback)` | [`verl/workers/sharding_manager/fsdp_ulysses.py`](../../verl/workers/sharding_manager/fsdp_ulysses.py#L45) |
| `FSDPUlyssesShardingManager.__init__` | `def(self, device_mesh)` | [`verl/workers/sharding_manager/fsdp_ulysses.py`](../../verl/workers/sharding_manager/fsdp_ulysses.py#L32) |
| `FSDPUlyssesShardingManager.postprocess_data` | `def(self, data)` | [`verl/workers/sharding_manager/fsdp_ulysses.py`](../../verl/workers/sharding_manager/fsdp_ulysses.py#L64) |
| `FSDPUlyssesShardingManager.preprocess_data` | `def(self, data)` | [`verl/workers/sharding_manager/fsdp_ulysses.py`](../../verl/workers/sharding_manager/fsdp_ulysses.py#L52) |
| `FSDPVLLMShardingManager.__enter__` | `def(self)` | [`verl/workers/sharding_manager/fsdp_vllm.py`](../../verl/workers/sharding_manager/fsdp_vllm.py#L114) |
| `FSDPVLLMShardingManager.__enter__.__collect_lora_params` | `def()` | [`verl/workers/sharding_manager/fsdp_vllm.py`](../../verl/workers/sharding_manager/fsdp_vllm.py#L115) |
| `FSDPVLLMShardingManager.__exit__` | `def(self, exc_type, exc_value, traceback)` | [`verl/workers/sharding_manager/fsdp_vllm.py`](../../verl/workers/sharding_manager/fsdp_vllm.py#L216) |
| `FSDPVLLMShardingManager.__init__` | `def(self, module, inference_engine, model_config, full_params, device_mesh, offload_param, load_format, layered_summon)` | [`verl/workers/sharding_manager/fsdp_vllm.py`](../../verl/workers/sharding_manager/fsdp_vllm.py#L55) |
| `FSDPVLLMShardingManager.postprocess_data` | `def(self, data)` | [`verl/workers/sharding_manager/fsdp_vllm.py`](../../verl/workers/sharding_manager/fsdp_vllm.py#L255) |
| `FSDPVLLMShardingManager.preprocess_data` | `def(self, data)` | [`verl/workers/sharding_manager/fsdp_vllm.py`](../../verl/workers/sharding_manager/fsdp_vllm.py#L237) |
| `FSDPVLLMShardingManager.update_params` | `def(self, updated_params, peft_config)` | [`verl/workers/sharding_manager/fsdp_vllm.py`](../../verl/workers/sharding_manager/fsdp_vllm.py#L262) |
| `FSDPVLLMShardingManager.update_params.replace_lora_wrapper` | `def(k)` | [`verl/workers/sharding_manager/fsdp_vllm.py`](../../verl/workers/sharding_manager/fsdp_vllm.py#L278) |
| `func_generator` | `def(self, method_name, dispatch_fn, collect_fn, execute_fn, blocking)` | [`verl/single_controller/ray/base.py`](../../verl/single_controller/ray/base.py#L44) |
| `func_generator.Functor.__call__` | `def(this, *args, **kwargs)` | [`verl/single_controller/ray/base.py`](../../verl/single_controller/ray/base.py#L46) |
| `FusedLinearForPPO.__init__` | `def(self, chunk_size)` | [`verl/utils/experimental/torch_functional.py`](../../verl/utils/experimental/torch_functional.py#L198) |
| `FusedLinearForPPO.forward` | `def(self, hidden_states, vocab_weights, input_ids, temperature)` | [`verl/utils/experimental/torch_functional.py`](../../verl/utils/experimental/torch_functional.py#L203) |
| `FusedLinearForPPOFunction.backward` | `def(ctx, dlog_probs, dentropy)` | [`verl/utils/experimental/torch_functional.py`](../../verl/utils/experimental/torch_functional.py#L132) |
| `FusedLinearForPPOFunction.forward` | `def(ctx, hidden_states, vocab_weights, input_ids, temperature, chunk_size)` | [`verl/utils/experimental/torch_functional.py`](../../verl/utils/experimental/torch_functional.py#L77) |
| `Gather.backward` | `def(ctx, grad_output)` | [`verl/utils/ulysses.py`](../../verl/utils/ulysses.py#L224) |
| `Gather.forward` | `def(ctx, group, local_tensor, gather_dim, grad_scaler, async_op)` | [`verl/utils/ulysses.py`](../../verl/utils/ulysses.py#L196) |
| `gather_from_labels` | `def(data, label)` | [`verl/utils/torch_functional.py`](../../verl/utils/torch_functional.py#L39) |
| `gather_heads_scatter_seq` | `def(x, head_dim, seq_dim, group)` | [`verl/utils/ulysses.py`](../../verl/utils/ulysses.py#L86) |
| `gather_outpus_and_unpad` | `def(x, gather_dim, unpad_dim, padding_size, grad_scaler, group)` | [`verl/utils/ulysses.py`](../../verl/utils/ulysses.py#L237) |
| `gather_seq_scatter_heads` | `def(x, seq_dim, head_dim, unpadded_dim_size, group)` | [`verl/utils/ulysses.py`](../../verl/utils/ulysses.py#L62) |
| `generate_attrs` | `def(corpus_by_cat, k, save_name)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/attributes/generate_attrs.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/attributes/generate_attrs.py#L140) |
| `generate_ground_truth_response` | `def(prompt)` | [`tests/e2e/envs/digit_completion/task.py`](../../tests/e2e/envs/digit_completion/task.py#L115) |
| `generate_hf_output` | `def(model, input_ids, attention_mask, tokenizer, max_response_length)` | [`tests/workers/rollout/utils_sglang.py`](../../tests/workers/rollout/utils_sglang.py#L105) |
| `generate_mturk_code` | `def(session_id)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/utils.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/utils.py#L43) |
| `generate_ngram_attrs` | `def(corpus_by_cat, ngram_range, k, attrs)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/attributes/generate_attrs.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/attributes/generate_attrs.py#L112) |
| `generate_product_prices` | `def(all_products)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/engine.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/engine.py#L180) |
| `generate_rl_dataset` | `def(target_hdfs_path_dir, local_dir)` | [`examples/data_preprocess/full_hh_rlhf.py`](../../examples/data_preprocess/full_hh_rlhf.py#L86) |
| `generate_rl_dataset.make_map_fn` | `def(split)` | [`examples/data_preprocess/full_hh_rlhf.py`](../../examples/data_preprocess/full_hh_rlhf.py#L93) |
| `generate_rl_dataset.make_map_fn.process_fn` | `def(example, idx)` | [`examples/data_preprocess/full_hh_rlhf.py`](../../examples/data_preprocess/full_hh_rlhf.py#L94) |
| `generate_rm_dataset` | `def(target_hdfs_path_dir, local_dir)` | [`examples/data_preprocess/full_hh_rlhf.py`](../../examples/data_preprocess/full_hh_rlhf.py#L58) |
| `generate_room` | `def(dim, p_change_directions, num_steps, num_boxes, tries, second_player, search_depth)` | [`agent_system/environments/env_package/sokoban/sokoban/room_utils.py`](../../agent_system/environments/env_package/sokoban/sokoban/room_utils.py#L227) |
| `generate_sft_dataset` | `def(target_hdfs_path_dir, local_dir)` | [`examples/data_preprocess/full_hh_rlhf.py`](../../examples/data_preprocess/full_hh_rlhf.py#L30) |
| `get_activation_offload_context` | `def(num_layers, model_layers, tensor_need_offloading_checker)` | [`verl/utils/activation_offload.py`](../../verl/utils/activation_offload.py#L394) |
| `get_activation_offload_context.group_prefetch_offload_commit_async` | `def(tensor)` | [`verl/utils/activation_offload.py`](../../verl/utils/activation_offload.py#L401) |
| `get_aggregated` | `def(output, lens, method)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/models/modules.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/models/modules.py#L18) |
| `get_attribute_reward` | `def(purchased_product, goal)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/goal.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/goal.py#L178) |
| `get_aux_metrics` | `def(self, test_proto)` | [`tests/ray_gpu/test_driverfunc_to_worker.py`](../../tests/ray_gpu/test_driverfunc_to_worker.py#L41) |
| `get_batch_logps` | `def(logits, labels, average_log_prob)` | [`recipe/spin/core_algos.py`](../../recipe/spin/core_algos.py#L160) |
| `get_checkpoint_tracker_filename` | `def(root_path)` | [`verl/utils/checkpoint/checkpoint_manager.py`](../../verl/utils/checkpoint/checkpoint_manager.py#L167) |
| `get_common_default_kwargs_for_parallel_linear` | `def()` | [`verl/utils/megatron/tensor_parallel.py`](../../verl/utils/megatron/tensor_parallel.py#L52) |
| `get_constant_schedule_with_warmup` | `def(optimizer, num_warmup_steps, last_epoch)` | [`verl/utils/torch_functional.py`](../../verl/utils/torch_functional.py#L505) |
| `get_constant_schedule_with_warmup.lr_lambda` | `def(current_step)` | [`verl/utils/torch_functional.py`](../../verl/utils/torch_functional.py#L522) |
| `get_corpus` | `def(products, keys, category_type)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/attributes/generate_attrs.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/attributes/generate_attrs.py#L76) |
| `get_cosine_schedule_with_warmup` | `def(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio, num_cycles, last_epoch)` | [`verl/utils/torch_functional.py`](../../verl/utils/torch_functional.py#L461) |
| `get_cosine_schedule_with_warmup.lr_lambda` | `def(current_step)` | [`verl/utils/torch_functional.py`](../../verl/utils/torch_functional.py#L495) |
| `get_custom_reward_fn` | `def(config)` | [`verl/trainer/ppo/reward.py`](../../verl/trainer/ppo/reward.py#L25) |
| `get_custom_reward_fn.wrapped_fn` | `def(*args, **kwargs)` | [`verl/trainer/ppo/reward.py`](../../verl/trainer/ppo/reward.py#L54) |
| `get_data` | `def(split, mem, filter_search)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/train_choice_il.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/train_choice_il.py#L104) |
| `get_data` | `def(split)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/train_search_il.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/train_search_il.py#L34) |
| `get_dataset` | `def(split, mem)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/train_choice_il.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/train_choice_il.py#L171) |
| `get_dataset` | `def(name, flip, variant, size)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/train_search_il.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/train_search_il.py#L72) |
| `get_default_kwargs_for_column_parallel_linear` | `def()` | [`verl/utils/megatron/tensor_parallel.py`](../../verl/utils/megatron/tensor_parallel.py#L63) |
| `get_default_kwargs_for_model_parallel_config` | `def()` | [`verl/utils/megatron/tensor_parallel.py`](../../verl/utils/megatron/tensor_parallel.py#L35) |
| `get_default_kwargs_for_parallel_embedding` | `def()` | [`verl/utils/megatron/tensor_parallel.py`](../../verl/utils/megatron/tensor_parallel.py#L84) |
| `get_default_kwargs_for_row_parallel_linear` | `def()` | [`verl/utils/megatron/tensor_parallel.py`](../../verl/utils/megatron/tensor_parallel.py#L79) |
| `get_default_model_parallel_config` | `def()` | [`verl/utils/megatron/tensor_parallel.py`](../../verl/utils/megatron/tensor_parallel.py#L46) |
| `get_device_flops` | `def(unit)` | [`verl/utils/flops_counter.py`](../../verl/utils/flops_counter.py#L41) |
| `get_device_flops.unit_convert` | `def(number, level)` | [`verl/utils/flops_counter.py`](../../verl/utils/flops_counter.py#L58) |
| `get_device_id` | `def()` | [`verl/utils/device.py`](../../verl/utils/device.py#L68) |
| `get_device_name` | `def()` | [`verl/utils/device.py`](../../verl/utils/device.py#L40) |
| `get_dir` | `def()` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L285) |
| `get_event_loop` | `def()` | [`verl/utils/ray_utils.py`](../../verl/utils/ray_utils.py#L85) |
| `get_fsdp_state_ctx` | `def(model, state_type, state_cfg, optim_cfg)` | [`verl/utils/fsdp_utils.py`](../../verl/utils/fsdp_utils.py#L387) |
| `get_fsdp_wrap_policy` | `def(module, config, is_lora)` | [`verl/utils/fsdp_utils.py`](../../verl/utils/fsdp_utils.py#L66) |
| `get_fsdp_wrap_policy._get_attr` | `def(attr_name, default_value)` | [`verl/utils/fsdp_utils.py`](../../verl/utils/fsdp_utils.py#L78) |
| `get_fsdp_wrap_policy.lambda_policy_fn` | `def(module)` | [`verl/utils/fsdp_utils.py`](../../verl/utils/fsdp_utils.py#L99) |
| `get_generation_config` | `def(model, trust_remote_code)` | [`verl/utils/model.py`](../../verl/utils/model.py#L72) |
| `get_goals` | `def(all_products, product_prices, human_goals)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/goal.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/goal.py#L16) |
| `get_gsm8k_data` | `def()` | [`tests/utils/gpu_tests/dataset/test_rl_dataset.py`](../../tests/utils/gpu_tests/dataset/test_rl_dataset.py#L21) |
| `get_gsm8k_data` | `def()` | [`tests/utils/gpu_tests/dataset/test_sft_dataset.py`](../../tests/utils/gpu_tests/dataset/test_sft_dataset.py#L20) |
| `get_hf_config_and_tokenizer_checkpoint_path` | `def(checkpoint_path)` | [`verl/utils/megatron_utils.py`](../../verl/utils/megatron_utils.py#L441) |
| `get_hf_model_checkpoint_path` | `def(checkpoint_path)` | [`verl/utils/megatron_utils.py`](../../verl/utils/megatron_utils.py#L436) |
| `get_huggingface_actor_config` | `def(model_name, override_config_kwargs, trust_remote_code)` | [`verl/utils/model.py`](../../verl/utils/model.py#L62) |
| `get_human_goals` | `def(all_products, product_prices)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/goal.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/goal.py#L22) |
| `get_image` | `def(card_name)` | [`agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/blackjack.py`](../../agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/blackjack.py#L12) |
| `get_image` | `def(card_name)` | [`agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/ezpoints.py`](../../agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/ezpoints.py#L12) |
| `get_image` | `def(card_name)` | [`agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/points.py`](../../agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/points.py#L12) |
| `get_init_weight_context_manager` | `def(use_meta_tensor, mesh)` | [`verl/utils/fsdp_utils.py`](../../verl/utils/fsdp_utils.py#L50) |
| `get_kl_controller` | `def(kl_ctrl)` | [`recipe/spin/core_algos.py`](../../recipe/spin/core_algos.py#L50) |
| `get_kl_controller` | `def(kl_ctrl)` | [`verl/trainer/ppo/core_algos.py`](../../verl/trainer/ppo/core_algos.py#L57) |
| `get_local_temp_path` | `def(hdfs_path, cache_dir)` | [`verl/utils/fs.py`](../../verl/utils/fs.py#L62) |
| `get_mcore_forward_fn` | `def(hf_config)` | [`verl/models/mcore/registry.py`](../../verl/models/mcore/registry.py#L235) |
| `get_mcore_forward_fused_fn` | `def(hf_config)` | [`verl/models/mcore/registry.py`](../../verl/models/mcore/registry.py#L253) |
| `get_mcore_forward_no_padding_fn` | `def(hf_config)` | [`verl/models/mcore/registry.py`](../../verl/models/mcore/registry.py#L244) |
| `get_mcore_weight_converter` | `def(hf_config, dtype)` | [`verl/models/mcore/registry.py`](../../verl/models/mcore/registry.py#L262) |
| `get_megatron_optimizer` | `def(model, config, no_weight_decay_cond, scale_lr_cond, lr_mult)` | [`verl/utils/megatron/optimizer.py`](../../verl/utils/megatron/optimizer.py#L20) |
| `get_model` | `def(model_provider_func, model_type, wrap_with_ddp, use_distributed_optimizer, transformer_config)` | [`verl/utils/megatron_utils.py`](../../verl/utils/megatron_utils.py#L45) |
| `get_model_checkpoint_path` | `def(checkpoint_path)` | [`verl/utils/megatron_utils.py`](../../verl/utils/megatron_utils.py#L431) |
| `get_model_config` | `def(model)` | [`verl/utils/megatron_utils.py`](../../verl/utils/megatron_utils.py#L41) |
| `get_model_size` | `def(model, scale)` | [`verl/utils/model.py`](../../verl/utils/model.py#L127) |
| `get_nccl_backend` | `def()` | [`verl/utils/device.py`](../../verl/utils/device.py#L76) |
| `get_nccl_id_store_by_name` | `def(name)` | [`verl/utils/rendezvous/ray_backend.py`](../../verl/utils/rendezvous/ray_backend.py#L32) |
| `get_obs_image` | `def(env)` | [`agent_system/environments/env_package/alfworld/envs.py`](../../agent_system/environments/env_package/alfworld/envs.py#L36) |
| `get_optimizer_checkpoint_path` | `def(checkpoint_path, use_distributed_optimizer)` | [`verl/utils/megatron_utils.py`](../../verl/utils/megatron_utils.py#L446) |
| `get_option_reward` | `def(purchased_options, goal_options)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/goal.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/goal.py#L209) |
| `get_parallel_gptmodel_from_config` | `def(tfconfig, hf_config, pre_process, post_process, share_embeddings_and_output_weights, value)` | [`verl/utils/model.py`](../../verl/utils/model.py#L419) |
| `get_parallel_model_from_config` | `def(config, megatron_config, pre_process, post_process, share_embeddings_and_output_weights, value)` | [`verl/utils/model.py`](../../verl/utils/model.py#L251) |
| `get_ppo_ray_runtime_env` | `def()` | [`verl/trainer/constants_ppo.py`](../../verl/trainer/constants_ppo.py#L38) |
| `get_predefined_dispatch_fn` | `def(dispatch_mode)` | [`verl/single_controller/base/decorator.py`](../../verl/single_controller/base/decorator.py#L443) |
| `get_predefined_execute_fn` | `def(execute_mode)` | [`verl/single_controller/base/decorator.py`](../../verl/single_controller/base/decorator.py#L466) |
| `get_product_per_page` | `def(top_n_products, page)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/engine.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/engine.py#L176) |
| `get_random_string` | `def(length)` | [`verl/single_controller/ray/base.py`](../../verl/single_controller/ray/base.py#L36) |
| `get_response_mask` | `def(response_id, eos_token, dtype)` | [`verl/utils/torch_functional.py`](../../verl/utils/torch_functional.py#L190) |
| `get_retriever` | `def(config)` | [`examples/search/retriever/retrieval_server.py`](../../examples/search/retriever/retrieval_server.py#L257) |
| `get_return_value` | `def(env, asin, options, search_terms, page_num, product)` | [`agent_system/environments/env_package/webshop/webshop/transfer/app.py`](../../agent_system/environments/env_package/webshop/webshop/transfer/app.py#L95) |
| `get_reverse_idx` | `def(idx_map)` | [`verl/utils/seqlen_balancing.py`](../../verl/utils/seqlen_balancing.py#L280) |
| `get_reward` | `def(purchased_product, goal, price, options, **kwargs)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/goal.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/goal.py#L228) |
| `get_rm_data` | `def()` | [`tests/utils/gpu_tests/dataset/test_rm_dataset.py`](../../tests/utils/gpu_tests/dataset/test_rm_dataset.py#L20) |
| `get_rng_states_checkpoint_path` | `def(checkpoint_path, only_rank0_save)` | [`verl/utils/megatron_utils.py`](../../verl/utils/megatron_utils.py#L458) |
| `get_rollout_config` | `def(max_response_length, max_prompt_length, dtype, tensor_parallel_size, tool_config_path)` | [`tests/workers/rollout/utils_sglang.py`](../../tests/workers/rollout/utils_sglang.py#L123) |
| `get_rope_index` | `def(processor, input_ids, image_grid_thw, video_grid_thw, second_per_grid_ts, attention_mask)` | [`verl/models/transformers/qwen2_vl.py`](../../verl/models/transformers/qwen2_vl.py#L64) |
| `get_rope_index` | `def(processor, input_ids, image_grid_thw, video_grid_thw, attention_mask, **kwargs)` | [`verl/models/transformers/qwen3_vl.py`](../../verl/models/transformers/qwen3_vl.py#L30) |
| `get_sandbox_fusion_messages` | `def()` | [`tests/workers/rollout/test_sglang_async_rollout_sf_tools.py`](../../tests/workers/rollout/test_sglang_async_rollout_sf_tools.py#L40) |
| `get_search_messages` | `def()` | [`tests/workers/rollout/test_sglang_async_rollout_search_tools.py`](../../tests/workers/rollout/test_sglang_async_rollout_search_tools.py#L50) |
| `get_seqlen_balanced_partitions` | `def(seqlen_list, k_partitions, equal_size)` | [`verl/utils/seqlen_balancing.py`](../../verl/utils/seqlen_balancing.py#L143) |
| `get_seqlen_balanced_partitions._check_and_sort_partitions` | `def(partitions)` | [`verl/utils/seqlen_balancing.py`](../../verl/utils/seqlen_balancing.py#L171) |
| `get_sharding_strategy` | `def(device_mesh)` | [`recipe/spin/fsdp_workers.py`](../../recipe/spin/fsdp_workers.py#L57) |
| `get_sharding_strategy` | `def(device_mesh)` | [`verl/workers/fsdp_workers.py`](../../verl/workers/fsdp_workers.py#L88) |
| `get_shortest_action_path` | `def(room_fixed, room_state, MAX_DEPTH)` | [`agent_system/environments/env_package/sokoban/sokoban/room_utils.py`](../../agent_system/environments/env_package/sokoban/sokoban/room_utils.py#L10) |
| `get_stop_words` | `def()` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/attributes/generate_attrs.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/attributes/generate_attrs.py#L20) |
| `get_supported_model` | `def(model_type)` | [`verl/models/mcore/registry.py`](../../verl/models/mcore/registry.py#L169) |
| `get_synthetic_goals` | `def(all_products, product_prices)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/goal.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/goal.py#L68) |
| `get_task_names` | `def(batch)` | [`verl/trainer/ppo/metric_utils.py`](../../verl/trainer/ppo/metric_utils.py#L74) |
| `get_tensor_parallel_partition_dim` | `def(param)` | [`verl/utils/megatron/tensor_parallel.py`](../../verl/utils/megatron/tensor_parallel.py#L99) |
| `get_tensor_parallel_partition_stride` | `def(param)` | [`verl/utils/megatron/tensor_parallel.py`](../../verl/utils/megatron/tensor_parallel.py#L104) |
| `get_tool_call_parser_type` | `def(tokenizer)` | [`verl/workers/rollout/sglang_rollout/sglang_rollout.py`](../../verl/workers/rollout/sglang_rollout/sglang_rollout.py#L107) |
| `get_top_attrs` | `def(attributes, k)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/attributes/generate_attrs.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/attributes/generate_attrs.py#L58) |
| `get_top_n_product_from_keywords` | `def(keywords, search_engine, all_products, product_item_dict, attribute_to_asins)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/engine.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/engine.py#L148) |
| `get_torch_device` | `def()` | [`verl/utils/device.py`](../../verl/utils/device.py#L55) |
| `get_train_val_env` | `def(env_class, config)` | [`agent_system/environments/env_package/sokoban/sokoban/utils.py`](../../agent_system/environments/env_package/sokoban/sokoban/utils.py#L56) |
| `get_trajectory_tracker` | `def()` | [`verl/utils/debug/trajectory_tracker.py`](../../verl/utils/debug/trajectory_tracker.py#L79) |
| `get_transformer_layer_offset` | `def(pipeline_rank, vp_rank, config)` | [`verl/utils/megatron_utils.py`](../../verl/utils/megatron_utils.py#L828) |
| `get_type_reward` | `def(purchased_product, goal)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/goal.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/goal.py#L130) |
| `get_ulysses_sequence_parallel_group` | `def()` | [`verl/utils/ulysses.py`](../../verl/utils/ulysses.py#L38) |
| `get_ulysses_sequence_parallel_rank` | `def(group)` | [`verl/utils/ulysses.py`](../../verl/utils/ulysses.py#L54) |
| `get_ulysses_sequence_parallel_world_size` | `def(group)` | [`verl/utils/ulysses.py`](../../verl/utils/ulysses.py#L46) |
| `get_unpad_data` | `def(attention_mask)` | [`verl/utils/torch_functional.py`](../../verl/utils/torch_functional.py#L577) |
| `get_version` | `def(pkg)` | [`verl/workers/rollout/vllm_rollout/__init__.py`](../../verl/workers/rollout/vllm_rollout/__init__.py#L20) |
| `get_visible_devices_keyword` | `def()` | [`verl/utils/device.py`](../../verl/utils/device.py#L32) |
| `get_weight_buffer_meta_from_module` | `def(module)` | [`verl/utils/memory_buffer.py`](../../verl/utils/memory_buffer.py#L60) |
| `get_weight_loader` | `def(arch)` | [`verl/models/weight_loader_registry.py`](../../verl/models/weight_loader_registry.py#L16) |
| `get_weight_saver` | `def(arch)` | [`verl/models/weight_loader_registry.py`](../../verl/models/weight_loader_registry.py#L29) |
| `get_wsd_schedule_with_warmup` | `def(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio, num_cycles, last_epoch, stable_ratio)` | [`verl/utils/torch_functional.py`](../../verl/utils/torch_functional.py#L589) |
| `get_wsd_schedule_with_warmup.lr_lambda` | `def(current_step)` | [`verl/utils/torch_functional.py`](../../verl/utils/torch_functional.py#L630) |
| `getkvs` | `def()` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L251) |
| `gptmodel_forward` | `def(model, input_ids, attention_mask, position_ids, sequence_parallel, value_model, pack_seqs, logits_processor, logits_processor_args)` | [`verl/models/mcore/model_forward.py`](../../verl/models/mcore/model_forward.py#L22) |
| `gptmodel_forward_qwen2_5_vl` | `def(*args, **kwargs)` | [`verl/models/mcore/model_forward.py`](../../verl/models/mcore/model_forward.py#L63) |
| `GPUMemoryLogger.__call__` | `def(self, decorated_function)` | [`verl/utils/debug/performance.py`](../../verl/utils/debug/performance.py#L78) |
| `GPUMemoryLogger.__call__.f` | `def(*args, **kwargs)` | [`verl/utils/debug/performance.py`](../../verl/utils/debug/performance.py#L79) |
| `GPUMemoryLogger.__init__` | `def(self, role, logger, level, log_only_rank_0)` | [`verl/utils/debug/performance.py`](../../verl/utils/debug/performance.py#L71) |
| `GPUMemoryLogger.log` | `def(self, func, *args, **kwargs)` | [`verl/utils/debug/performance.py`](../../verl/utils/debug/performance.py#L84) |
| `grade_answer` | `def(given_answer, ground_truth)` | [`verl/utils/reward_score/prime_math/__init__.py`](../../verl/utils/reward_score/prime_math/__init__.py#L241) |
| `greedy_partition` | `def(seqlen_list, k_partitions, equal_size)` | [`verl/utils/seqlen_balancing.py`](../../verl/utils/seqlen_balancing.py#L125) |
| `GroupCommitFunction.backward` | `def(ctx, grad_output)` | [`verl/utils/activation_offload.py`](../../verl/utils/activation_offload.py#L120) |
| `GroupCommitFunction.forward` | `def(ctx, tensor, cpu_offload_handler)` | [`verl/utils/activation_offload.py`](../../verl/utils/activation_offload.py#L112) |
| `Gsm8kTool.__init__` | `def(self, config, tool_schema)` | [`verl/tools/gsm8k_tool.py`](../../verl/tools/gsm8k_tool.py#L40) |
| `Gsm8kTool.calc_reward` | `async def(self, instance_id, **kwargs)` | [`verl/tools/gsm8k_tool.py`](../../verl/tools/gsm8k_tool.py#L94) |
| `Gsm8kTool.create` | `async def(self, instance_id, ground_truth, **kwargs)` | [`verl/tools/gsm8k_tool.py`](../../verl/tools/gsm8k_tool.py#L66) |
| `Gsm8kTool.execute` | `async def(self, instance_id, parameters, **kwargs)` | [`verl/tools/gsm8k_tool.py`](../../verl/tools/gsm8k_tool.py#L76) |
| `Gsm8kTool.get_openai_tool_schema` | `def(self)` | [`verl/tools/gsm8k_tool.py`](../../verl/tools/gsm8k_tool.py#L63) |
| `Gsm8kTool.release` | `async def(self, instance_id, **kwargs)` | [`verl/tools/gsm8k_tool.py`](../../verl/tools/gsm8k_tool.py#L103) |
| `gym_projection` | `def(text_actions, env_name)` | [`agent_system/environments/env_package/gym_cards/projection.py`](../../agent_system/environments/env_package/gym_cards/projection.py#L20) |
| `GymCardEnvironmentManager.__init__` | `def(self, envs, projection_f, config)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L344) |
| `GymCardEnvironmentManager.__init__` | `def(self, envs, projection_f, config)` | [`recipe/hgpo/env_manager.py`](../../recipe/hgpo/env_manager.py#L331) |
| `GymCardEnvironmentManager.build_text_obs` | `def(self, infos)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L364) |
| `GymCardEnvironmentManager.build_text_obs` | `def(self, infos)` | [`recipe/hgpo/env_manager.py`](../../recipe/hgpo/env_manager.py#L351) |
| `GymCardEnvironmentManager.reset` | `def(self, kwargs)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L347) |
| `GymCardEnvironmentManager.reset` | `def(self, kwargs)` | [`recipe/hgpo/env_manager.py`](../../recipe/hgpo/env_manager.py#L334) |
| `GymCardEnvironmentManager.step` | `def(self, text_actions)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L354) |
| `GymCardEnvironmentManager.step` | `def(self, text_actions)` | [`recipe/hgpo/env_manager.py`](../../recipe/hgpo/env_manager.py#L341) |
| `GymCardsWorker.__init__` | `def(self, env_id)` | [`agent_system/environments/env_package/gym_cards/envs.py`](../../agent_system/environments/env_package/gym_cards/envs.py#L28) |
| `GymCardsWorker.reset` | `def(self, seed_for_reset)` | [`agent_system/environments/env_package/gym_cards/envs.py`](../../agent_system/environments/env_package/gym_cards/envs.py#L46) |
| `GymCardsWorker.step` | `def(self, action)` | [`agent_system/environments/env_package/gym_cards/envs.py`](../../agent_system/environments/env_package/gym_cards/envs.py#L41) |
| `GymMultiProcessEnv.__del__` | `def(self)` | [`agent_system/environments/env_package/gym_cards/envs.py`](../../agent_system/environments/env_package/gym_cards/envs.py#L159) |
| `GymMultiProcessEnv.__init__` | `def(self, env_id, seed, env_num, group_n, resources_per_worker, is_train)` | [`agent_system/environments/env_package/gym_cards/envs.py`](../../agent_system/environments/env_package/gym_cards/envs.py#L64) |
| `GymMultiProcessEnv.close` | `def(self)` | [`agent_system/environments/env_package/gym_cards/envs.py`](../../agent_system/environments/env_package/gym_cards/envs.py#L151) |
| `GymMultiProcessEnv.reset` | `def(self)` | [`agent_system/environments/env_package/gym_cards/envs.py`](../../agent_system/environments/env_package/gym_cards/envs.py#L119) |
| `GymMultiProcessEnv.step` | `def(self, actions)` | [`agent_system/environments/env_package/gym_cards/envs.py`](../../agent_system/environments/env_package/gym_cards/envs.py#L92) |
| `HackSelf.__init__` | `def(self)` | [`tests/ray_gpu/test_driverfunc_to_worker.py`](../../tests/ray_gpu/test_driverfunc_to_worker.py#L37) |
| `handle_base` | `def(x)` | [`verl/utils/reward_score/prime_math/grader.py`](../../verl/utils/reward_score/prime_math/grader.py#L140) |
| `handle_pi` | `def(string, pi)` | [`verl/utils/reward_score/prime_math/grader.py`](../../verl/utils/reward_score/prime_math/grader.py#L149) |
| `hf_processor` | `def(name_or_path, **kwargs)` | [`verl/utils/tokenizer.py`](../../verl/utils/tokenizer.py#L64) |
| `hf_to_mcore_config` | `def(hf_config, dtype, **override_transformer_config_kwargs)` | [`verl/models/mcore/registry.py`](../../verl/models/mcore/registry.py#L179) |
| `hf_to_mcore_config_dense` | `def(hf_config, dtype, **override_transformer_config_kwargs)` | [`verl/models/mcore/config_converter.py`](../../verl/models/mcore/config_converter.py#L165) |
| `hf_to_mcore_config_dpskv3` | `def(hf_config, dtype, **override_transformer_config_kwargs)` | [`verl/models/mcore/config_converter.py`](../../verl/models/mcore/config_converter.py#L284) |
| `hf_to_mcore_config_llama4` | `def(hf_config, dtype, **override_transformer_config_kwargs)` | [`verl/models/mcore/config_converter.py`](../../verl/models/mcore/config_converter.py#L384) |
| `hf_to_mcore_config_mixtral` | `def(hf_config, dtype, **override_transformer_config_kwargs)` | [`verl/models/mcore/config_converter.py`](../../verl/models/mcore/config_converter.py#L219) |
| `hf_to_mcore_config_qwen2_5_vl` | `def(hf_config, dtype, **override_transformer_config_kwargs)` | [`verl/models/mcore/config_converter.py`](../../verl/models/mcore/config_converter.py#L365) |
| `hf_to_mcore_config_qwen2moe` | `def(hf_config, dtype, **override_transformer_config_kwargs)` | [`verl/models/mcore/config_converter.py`](../../verl/models/mcore/config_converter.py#L185) |
| `hf_to_mcore_config_qwen3moe` | `def(hf_config, dtype, **override_transformer_config_kwargs)` | [`verl/models/mcore/config_converter.py`](../../verl/models/mcore/config_converter.py#L252) |
| `hf_tokenizer` | `def(name_or_path, correct_pad_token, correct_gemma2, **kwargs)` | [`verl/utils/tokenizer.py`](../../verl/utils/tokenizer.py#L36) |
| `HFRollout.__init__` | `def(self, module, config)` | [`verl/workers/rollout/hf_rollout.py`](../../verl/workers/rollout/hf_rollout.py#L40) |
| `HFRollout._generate_minibatch` | `def(self, prompts)` | [`verl/workers/rollout/hf_rollout.py`](../../verl/workers/rollout/hf_rollout.py#L54) |
| `HFRollout.generate_sequences` | `def(self, prompts)` | [`verl/workers/rollout/hf_rollout.py`](../../verl/workers/rollout/hf_rollout.py#L45) |
| `hgpo_advantage_estimate` | `def(token_level_rewards, step_rewards, response_mask, anchor_obs, index, traj_index, history_length, epsilon, mode, length_weight_alpha, base_group)` | [`recipe/hgpo/core_hgpo.py`](../../recipe/hgpo/core_hgpo.py#L146) |
| `hgpo_advantage_estimate.aggregate_items` | `def(items)` | [`recipe/hgpo/core_hgpo.py`](../../recipe/hgpo/core_hgpo.py#L183) |
| `home` | `def()` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/app.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/app.py#L46) |
| `HumanOutputFormat.__init__` | `def(self, filename_or_file)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L31) |
| `HumanOutputFormat._truncate` | `def(self, s)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L74) |
| `HumanOutputFormat.close` | `def(self)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L86) |
| `HumanOutputFormat.writekvs` | `def(self, kvs)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L40) |
| `HumanOutputFormat.writeseq` | `def(self, seq)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L77) |
| `HumanPolicy.__init__` | `def(self)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/models/models.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/models/models.py#L34) |
| `HumanPolicy.forward` | `def(self, observation, available_actions)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/models/models.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/models/models.py#L37) |
| `HybridEngineBaseTokenizer.all_special_ids` | `def(self)` | [`verl/workers/rollout/tokenizer.py`](../../verl/workers/rollout/tokenizer.py#L57) |
| `HybridEngineBaseTokenizer.all_special_tokens` | `def(self)` | [`verl/workers/rollout/tokenizer.py`](../../verl/workers/rollout/tokenizer.py#L65) |
| `HybridEngineBaseTokenizer.convert_ids_to_tokens` | `def(self, ids, skip_special_tokens)` | [`verl/workers/rollout/tokenizer.py`](../../verl/workers/rollout/tokenizer.py#L120) |
| `HybridEngineBaseTokenizer.convert_tokens_to_string` | `def(self, tokens)` | [`verl/workers/rollout/tokenizer.py`](../../verl/workers/rollout/tokenizer.py#L149) |
| `HybridEngineBaseTokenizer.decode` | `def(self, token_ids, skip_special_tokens, clean_up_tokenization_spaces, **kwargs)` | [`verl/workers/rollout/tokenizer.py`](../../verl/workers/rollout/tokenizer.py#L90) |
| `HybridEngineBaseTokenizer.encode` | `def(self, text)` | [`verl/workers/rollout/tokenizer.py`](../../verl/workers/rollout/tokenizer.py#L74) |
| `HybridEngineBaseTokenizer.eos_token_id` | `def(self)` | [`verl/workers/rollout/tokenizer.py`](../../verl/workers/rollout/tokenizer.py#L48) |
| `HybridEngineBaseTokenizer.get_added_vocab` | `def(self)` | [`verl/workers/rollout/tokenizer.py`](../../verl/workers/rollout/tokenizer.py#L137) |
| `HybridEngineBaseTokenizer.is_fast` | `def(self)` | [`verl/workers/rollout/tokenizer.py`](../../verl/workers/rollout/tokenizer.py#L163) |
| `HybridEngineBaseTokenizer.pad_token_id` | `def(self)` | [`verl/workers/rollout/tokenizer.py`](../../verl/workers/rollout/tokenizer.py#L40) |
| `HybridEngineBaseTokenizer.vocab_size` | `def(self)` | [`verl/workers/rollout/tokenizer.py`](../../verl/workers/rollout/tokenizer.py#L32) |
| `HybridWorker.foo` | `def(self, x)` | [`tests/ray_cpu/test_fused_workers.py`](../../tests/ray_cpu/test_fused_workers.py#L54) |
| `hydra_entry` | `def(cfg)` | [`tests/e2e/sft/test_sp_loss_match.py`](../../tests/e2e/sft/test_sp_loss_match.py#L134) |
| `import_external_libs` | `def(external_libs)` | [`verl/utils/import_utils.py`](../../verl/utils/import_utils.py#L72) |
| `index` | `def(session_id, **kwargs)` | [`agent_system/environments/env_package/webshop/webshop/transfer/webshop_lite.py`](../../agent_system/environments/env_package/webshop/webshop/transfer/webshop_lite.py#L21) |
| `index` | `def(session_id)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/app.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/app.py#L50) |
| `info` | `def(*args)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L266) |
| `info_to_text_obs` | `def(env_name, info)` | [`agent_system/environments/env_package/gym_cards/gym-cards/text_wrapper.py`](../../agent_system/environments/env_package/gym_cards/gym-cards/text_wrapper.py#L4) |
| `init_async_rollout_manager` | `def(config, scheduler_kwargs)` | [`tests/workers/rollout/async_rollout_utils.py`](../../tests/workers/rollout/async_rollout_utils.py#L27) |
| `init_config` | `def()` | [`tests/workers/rollout/test_vllm_multi_turn.py`](../../tests/workers/rollout/test_vllm_multi_turn.py#L28) |
| `init_execution_pool` | `def(num_workers, enable_global_rate_limit, rate_limit, mode)` | [`verl/tools/sandbox_fusion_tools.py`](../../verl/tools/sandbox_fusion_tools.py#L88) |
| `init_fn` | `def(x)` | [`verl/utils/fsdp_utils.py`](../../verl/utils/fsdp_utils.py#L43) |
| `init_mcore_model` | `def(tfconfig, hf_config, pre_process, post_process, **extra_kwargs)` | [`verl/models/mcore/registry.py`](../../verl/models/mcore/registry.py#L197) |
| `init_megatron_optim_config` | `def(optim_config)` | [`verl/utils/megatron_utils.py`](../../verl/utils/megatron_utils.py#L201) |
| `init_predefined_dispatch_mode` | `def()` | [`verl/single_controller/base/decorator.py`](../../verl/single_controller/base/decorator.py#L39) |
| `init_predefined_execute_mode` | `def()` | [`verl/single_controller/base/decorator.py`](../../verl/single_controller/base/decorator.py#L67) |
| `init_ray` | `def()` | [`tests/ray_cpu/test_ray_utils.py`](../../tests/ray_cpu/test_ray_utils.py#L23) |
| `init_search_engine` | `def(num_products)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/engine.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/engine.py#L195) |
| `init_search_execution_pool` | `def(num_workers, enable_global_rate_limit, rate_limit, mode)` | [`verl/tools/search_tool.py`](../../verl/tools/search_tool.py#L102) |
| `initialize_global_process_group` | `def(timeout_second, spmd)` | [`tests/workers/rollout/utils_sglang.py`](../../tests/workers/rollout/utils_sglang.py#L55) |
| `initialize_global_process_group` | `def(timeout_second)` | [`verl/utils/distributed.py`](../../verl/utils/distributed.py#L20) |
| `is_bust` | `def(hand)` | [`agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/blackjack.py`](../../agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/blackjack.py#L66) |
| `is_correct_minerva` | `def(solution_str, gt, gt_need_extract, answer_pattern)` | [`verl/utils/reward_score/math_dapo.py`](../../verl/utils/reward_score/math_dapo.py#L166) |
| `is_correct_strict_box` | `def(pred, gt, pause_tokens_index)` | [`verl/utils/reward_score/math_dapo.py`](../../verl/utils/reward_score/math_dapo.py#L192) |
| `is_digit` | `def(s)` | [`verl/utils/reward_score/prime_math/grader.py`](../../verl/utils/reward_score/prime_math/grader.py#L110) |
| `is_equiv` | `def(str1, str2, verbose)` | [`verl/utils/reward_score/math.py`](../../verl/utils/reward_score/math.py#L32) |
| `is_ipv4` | `def(ip_str)` | [`verl/utils/net_utils.py`](../../verl/utils/net_utils.py#L30) |
| `is_ipv6` | `def(ip_str)` | [`verl/utils/net_utils.py`](../../verl/utils/net_utils.py#L47) |
| `is_megatron_core_available` | `def()` | [`verl/utils/import_utils.py`](../../verl/utils/import_utils.py#L28) |
| `is_natural` | `def(hand)` | [`agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/blackjack.py`](../../agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/blackjack.py#L75) |
| `is_non_local` | `def(path)` | [`verl/utils/fs.py`](../../verl/utils/fs.py#L35) |
| `is_nvtx_available` | `def()` | [`verl/utils/import_utils.py`](../../verl/utils/import_utils.py#L55) |
| `is_sequence_parallel_param` | `def(param)` | [`verl/utils/megatron/sequence_parallel.py`](../../verl/utils/megatron/sequence_parallel.py#L25) |
| `is_sglang_available` | `def()` | [`verl/utils/import_utils.py`](../../verl/utils/import_utils.py#L46) |
| `is_tensor_parallel_param` | `def(param)` | [`verl/utils/megatron/tensor_parallel.py`](../../verl/utils/megatron/tensor_parallel.py#L95) |
| `is_torch_npu_available` | `def()` | [`verl/utils/device.py`](../../verl/utils/device.py#L18) |
| `is_transformers_version_in_range` | `def(min_version, max_version)` | [`verl/utils/transformers_compat.py`](../../verl/utils/transformers_compat.py#L40) |
| `is_trl_available` | `def()` | [`verl/utils/import_utils.py`](../../verl/utils/import_utils.py#L64) |
| `is_version_ge` | `def(pkg, minver)` | [`verl/utils/vllm_utils.py`](../../verl/utils/vllm_utils.py#L244) |
| `is_vllm_available` | `def()` | [`verl/utils/import_utils.py`](../../verl/utils/import_utils.py#L37) |
| `item_page` | `def(session_id, asin, keywords, page, options)` | [`agent_system/environments/env_package/webshop/webshop/transfer/webshop_lite.py`](../../agent_system/environments/env_package/webshop/webshop/transfer/webshop_lite.py#L39) |
| `item_page` | `def(session_id, asin, keywords, page, options)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/app.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/app.py#L150) |
| `item_sub_page` | `def(session_id, asin, keywords, page, sub_page, options)` | [`agent_system/environments/env_package/webshop/webshop/transfer/webshop_lite.py`](../../agent_system/environments/env_package/webshop/webshop/transfer/webshop_lite.py#L54) |
| `item_sub_page` | `def(session_id, asin, keywords, page, sub_page, options)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/app.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/app.py#L187) |
| `iter_task_row_masks` | `def(task_ids, task_id_names)` | [`verl/trainer/ppo/metric_utils.py`](../../verl/trainer/ppo/metric_utils.py#L112) |
| `JSONOutputFormat.__init__` | `def(self, filename)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L92) |
| `JSONOutputFormat.close` | `def(self)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L103) |
| `JSONOutputFormat.writekvs` | `def(self, kvs)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L95) |
| `karmarkar_karp` | `def(seqlen_list, k_partitions, equal_size)` | [`verl/utils/seqlen_balancing.py`](../../verl/utils/seqlen_balancing.py#L23) |
| `karmarkar_karp.Set.__init__` | `def(self)` | [`verl/utils/seqlen_balancing.py`](../../verl/utils/seqlen_balancing.py#L26) |
| `karmarkar_karp.Set.__lt__` | `def(self, other)` | [`verl/utils/seqlen_balancing.py`](../../verl/utils/seqlen_balancing.py#L39) |
| `karmarkar_karp.Set.add` | `def(self, idx, val)` | [`verl/utils/seqlen_balancing.py`](../../verl/utils/seqlen_balancing.py#L30) |
| `karmarkar_karp.Set.merge` | `def(self, other)` | [`verl/utils/seqlen_balancing.py`](../../verl/utils/seqlen_balancing.py#L34) |
| `karmarkar_karp.State.__init__` | `def(self, items, k)` | [`verl/utils/seqlen_balancing.py`](../../verl/utils/seqlen_balancing.py#L47) |
| `karmarkar_karp.State.__lt__` | `def(self, other)` | [`verl/utils/seqlen_balancing.py`](../../verl/utils/seqlen_balancing.py#L74) |
| `karmarkar_karp.State.__repr__` | `def(self)` | [`verl/utils/seqlen_balancing.py`](../../verl/utils/seqlen_balancing.py#L82) |
| `karmarkar_karp.State.get_partitions` | `def(self)` | [`verl/utils/seqlen_balancing.py`](../../verl/utils/seqlen_balancing.py#L56) |
| `karmarkar_karp.State.merge` | `def(self, other)` | [`verl/utils/seqlen_balancing.py`](../../verl/utils/seqlen_balancing.py#L65) |
| `karmarkar_karp.State.spread` | `def(self)` | [`verl/utils/seqlen_balancing.py`](../../verl/utils/seqlen_balancing.py#L71) |
| `kl_penalty` | `def(logprob, ref_logprob, kl_penalty)` | [`verl/trainer/ppo/core_algos.py`](../../verl/trainer/ppo/core_algos.py#L641) |
| `KVWriter.writekvs` | `def(self, kvs)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L21) |
| `LambdaLayer.__init__` | `def(self, fn)` | [`verl/utils/model.py`](../../verl/utils/model.py#L37) |
| `LambdaLayer.forward` | `def(self, *args, **kwargs)` | [`verl/utils/model.py`](../../verl/utils/model.py#L41) |
| `last_boxed_only_string` | `def(string)` | [`verl/utils/reward_score/math.py`](../../verl/utils/reward_score/math.py#L63) |
| `last_boxed_only_string` | `def(string)` | [`verl/utils/reward_score/math_dapo.py`](../../verl/utils/reward_score/math_dapo.py#L20) |
| `layered_summon_lora_params` | `def(fsdp_module)` | [`verl/utils/fsdp_utils.py`](../../verl/utils/fsdp_utils.py#L462) |
| `layered_summon_lora_params.__prefix_submodules` | `def(module, prefix)` | [`verl/utils/fsdp_utils.py`](../../verl/utils/fsdp_utils.py#L465) |
| `levenshtein` | `def(s1, s2)` | [`tests/workers/rollout/test_vllm_hf_loader.py`](../../tests/workers/rollout/test_vllm_hf_loader.py#L26) |
| `levenshtein` | `def(s1, s2)` | [`tests/workers/rollout/test_vllm_spmd.py`](../../tests/workers/rollout/test_vllm_spmd.py#L28) |
| `levenshtein` | `def(s1, s2)` | [`tests/workers/rollout/utils_sglang.py`](../../tests/workers/rollout/utils_sglang.py#L26) |
| `LinearForLastLayer.__init__` | `def(self, input_size, output_size)` | [`verl/models/llama/megatron/layers/parallel_linear.py`](../../verl/models/llama/megatron/layers/parallel_linear.py#L83) |
| `LinearForLastLayer.forward` | `def(self, input_, weight, runtime_gather_output)` | [`verl/models/llama/megatron/layers/parallel_linear.py`](../../verl/models/llama/megatron/layers/parallel_linear.py#L96) |
| `list_of_dict_to_dict_of_list` | `def(list_of_dict)` | [`verl/protocol.py`](../../verl/protocol.py#L128) |
| `llama_attn_forward` | `def(self, hidden_states, position_embeddings, attention_mask, past_key_value, cache_position, **kwargs)` | [`verl/models/transformers/llama.py`](../../verl/models/transformers/llama.py#L165) |
| `llama_flash_attn_forward` | `def(self, hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, cache_position, position_embeddings, **kwargs)` | [`verl/models/transformers/llama.py`](../../verl/models/transformers/llama.py#L42) |
| `LlamaDynamicNTKScalingRotaryEmbedding.__init__` | `def(self, dim, max_position_embeddings, base, device, scaling_factor)` | [`verl/models/llama/megatron/layers/parallel_attention.py`](../../verl/models/llama/megatron/layers/parallel_attention.py#L94) |
| `LlamaDynamicNTKScalingRotaryEmbedding._set_cos_sin_cache` | `def(self, seq_len, device, dtype)` | [`verl/models/llama/megatron/layers/parallel_attention.py`](../../verl/models/llama/megatron/layers/parallel_attention.py#L98) |
| `LlamaLinearScalingRotaryEmbedding.__init__` | `def(self, dim, max_position_embeddings, base, device, scaling_factor)` | [`verl/models/llama/megatron/layers/parallel_attention.py`](../../verl/models/llama/megatron/layers/parallel_attention.py#L75) |
| `LlamaLinearScalingRotaryEmbedding._set_cos_sin_cache` | `def(self, seq_len, device, dtype)` | [`verl/models/llama/megatron/layers/parallel_attention.py`](../../verl/models/llama/megatron/layers/parallel_attention.py#L79) |
| `LlamaLlama3ScalingRotaryEmbedding.__init__` | `def(self, dim, config, max_position_embeddings, base, device)` | [`verl/models/llama/megatron/layers/parallel_attention.py`](../../verl/models/llama/megatron/layers/parallel_attention.py#L116) |
| `LlamaRotaryEmbedding.__init__` | `def(self, dim, max_position_embeddings, base, device)` | [`verl/models/llama/megatron/layers/parallel_attention.py`](../../verl/models/llama/megatron/layers/parallel_attention.py#L39) |
| `LlamaRotaryEmbedding._set_cos_sin_cache` | `def(self, seq_len, device, dtype)` | [`verl/models/llama/megatron/layers/parallel_attention.py`](../../verl/models/llama/megatron/layers/parallel_attention.py#L51) |
| `LlamaRotaryEmbedding.forward` | `def(self, x, seq_len)` | [`verl/models/llama/megatron/layers/parallel_attention.py`](../../verl/models/llama/megatron/layers/parallel_attention.py#L61) |
| `load_available_ports` | `def(port_file)` | [`agent_system/environments/env_package/appworld/envs.py`](../../agent_system/environments/env_package/appworld/envs.py#L23) |
| `load_config_file` | `def(path)` | [`agent_system/environments/env_package/alfworld/envs.py`](../../agent_system/environments/env_package/alfworld/envs.py#L30) |
| `load_corpus` | `def(corpus_path)` | [`examples/search/retriever/retrieval_server.py`](../../examples/search/retriever/retrieval_server.py#L17) |
| `load_docs` | `def(corpus, doc_idxs)` | [`examples/search/retriever/retrieval_server.py`](../../examples/search/retriever/retrieval_server.py#L30) |
| `load_expectations` | `def(expect_file)` | [`verl/utils/expected_config.py`](../../verl/utils/expected_config.py#L70) |
| `load_expectations._flatten` | `def(prefix, node)` | [`verl/utils/expected_config.py`](../../verl/utils/expected_config.py#L82) |
| `load_extern_type` | `def(file_path, type_name)` | [`verl/utils/import_utils.py`](../../verl/utils/import_utils.py#L83) |
| `load_fsdp2_model_to_gpu` | `def(model)` | [`verl/utils/fsdp_utils.py`](../../verl/utils/fsdp_utils.py#L181) |
| `load_fsdp_model_to_gpu` | `def(model)` | [`verl/utils/fsdp_utils.py`](../../verl/utils/fsdp_utils.py#L161) |
| `load_fsdp_optimizer` | `def(optimizer, device_id)` | [`verl/utils/fsdp_utils.py`](../../verl/utils/fsdp_utils.py#L200) |
| `load_mcore_dist_weights` | `def(parallel_model, dist_weight_path, is_value_model)` | [`verl/utils/model.py`](../../verl/utils/model.py#L398) |
| `load_megatron_copy_params` | `def(optimizers)` | [`verl/utils/megatron_utils.py`](../../verl/utils/megatron_utils.py#L344) |
| `load_megatron_copy_params._iter_opts` | `def(opt)` | [`verl/utils/megatron_utils.py`](../../verl/utils/megatron_utils.py#L352) |
| `load_megatron_copy_params.load_group_to_gpu` | `def(group)` | [`verl/utils/megatron_utils.py`](../../verl/utils/megatron_utils.py#L363) |
| `load_megatron_copy_params.load_tensor_to_gpu` | `def(tensor)` | [`verl/utils/megatron_utils.py`](../../verl/utils/megatron_utils.py#L357) |
| `load_megatron_gptmodel_weights` | `def(config, model_config, parallel_model, params_dtype, is_value_model, local_cache_path)` | [`verl/utils/model.py`](../../verl/utils/model.py#L348) |
| `load_megatron_model_to_gpu` | `def(models, load_grad)` | [`verl/utils/megatron_utils.py`](../../verl/utils/megatron_utils.py#L276) |
| `load_megatron_model_weights` | `def(config, model_config, parallel_model, params_dtype, is_value_model, local_cache_path)` | [`verl/utils/model.py`](../../verl/utils/model.py#L327) |
| `load_megatron_optimizer` | `def(optimizers)` | [`verl/utils/megatron_utils.py`](../../verl/utils/megatron_utils.py#L404) |
| `load_megatron_optimizer._iter_opts` | `def(opt)` | [`verl/utils/megatron_utils.py`](../../verl/utils/megatron_utils.py#L405) |
| `load_model` | `def(model_path, use_fp16)` | [`examples/search/retriever/retrieval_server.py`](../../examples/search/retriever/retrieval_server.py#L35) |
| `load_products` | `def(num)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/attributes/generate_attrs.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/attributes/generate_attrs.py#L26) |
| `load_products` | `def(filepath, attrpath, num_products, human_goals)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/engine.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/engine.py#L230) |
| `load_reward_manager` | `def(config, tokenizer, num_examine, **reward_kwargs)` | [`verl/trainer/ppo/reward.py`](../../verl/trainer/ppo/reward.py#L60) |
| `load_skill_content` | `def(skills_dir, skill_mapping)` | [`verl/trainer/ppo/rlsd_utils.py`](../../verl/trainer/ppo/rlsd_utils.py#L24) |
| `load_skill_mapping` | `def(skills_dir)` | [`verl/trainer/ppo/rlsd_utils.py`](../../verl/trainer/ppo/rlsd_utils.py#L18) |
| `load_state_dict_to_megatron_gptmodel` | `def(state_dict, wrapped_models, config, params_dtype, is_value_model)` | [`verl/models/mcore/loader.py`](../../verl/models/mcore/loader.py#L52) |
| `load_state_dict_to_megatron_gptmodel._broadcast_tensor` | `def(tensor, name)` | [`verl/models/mcore/loader.py`](../../verl/models/mcore/loader.py#L97) |
| `load_state_dict_to_megatron_gptmodel._broadcast_tp_shard_tensor` | `def(tensor, name, chunk_dim, mutate_func)` | [`verl/models/mcore/loader.py`](../../verl/models/mcore/loader.py#L177) |
| `load_state_dict_to_megatron_gptmodel._broadcast_tp_shard_tensor_gate_up` | `def(tensor, gate_name, up_name)` | [`verl/models/mcore/loader.py`](../../verl/models/mcore/loader.py#L222) |
| `load_state_dict_to_megatron_gptmodel._broadcast_tp_shard_tensor_qkv` | `def(tensor, q_name, k_name, v_name, bias)` | [`verl/models/mcore/loader.py`](../../verl/models/mcore/loader.py#L270) |
| `load_state_dict_to_megatron_gptmodel._broadcast_tp_shard_tensor_vocab` | `def(tensor, name, chunk_dim, mutate_func)` | [`verl/models/mcore/loader.py`](../../verl/models/mcore/loader.py#L131) |
| `load_state_dict_to_megatron_gptmodel._get_gpt_model` | `def(model)` | [`verl/models/mcore/loader.py`](../../verl/models/mcore/loader.py#L63) |
| `load_state_dict_to_megatron_gptmodel.broadcast_params` | `def(module)` | [`verl/models/mcore/loader.py`](../../verl/models/mcore/loader.py#L66) |
| `load_state_dict_to_megatron_llama` | `def(state_dict, wrapped_models, config, params_dtype, is_value_model, tie_word_embeddings)` | [`verl/models/llama/megatron/checkpoint_utils/llama_loader.py`](../../verl/models/llama/megatron/checkpoint_utils/llama_loader.py#L51) |
| `load_state_dict_to_megatron_llama` | `def(state_dict, wrapped_models, config, params_dtype, is_value_model, tie_word_embeddings)` | [`verl/models/llama/megatron/checkpoint_utils/llama_loader_depracated.py`](../../verl/models/llama/megatron/checkpoint_utils/llama_loader_depracated.py#L51) |
| `load_state_dict_to_megatron_llama._broadcast_tensor` | `def(tensor, name)` | [`verl/models/llama/megatron/checkpoint_utils/llama_loader_depracated.py`](../../verl/models/llama/megatron/checkpoint_utils/llama_loader_depracated.py#L94) |
| `load_state_dict_to_megatron_llama._broadcast_tp_shard_tensor` | `def(tensor, name, chunk_dim, mutate_func)` | [`verl/models/llama/megatron/checkpoint_utils/llama_loader_depracated.py`](../../verl/models/llama/megatron/checkpoint_utils/llama_loader_depracated.py#L174) |
| `load_state_dict_to_megatron_llama._broadcast_tp_shard_tensor_gate_up` | `def(tensor, gate_name, up_name)` | [`verl/models/llama/megatron/checkpoint_utils/llama_loader_depracated.py`](../../verl/models/llama/megatron/checkpoint_utils/llama_loader_depracated.py#L219) |
| `load_state_dict_to_megatron_llama._broadcast_tp_shard_tensor_qkv` | `def(tensor, q_name, k_name, v_name)` | [`verl/models/llama/megatron/checkpoint_utils/llama_loader_depracated.py`](../../verl/models/llama/megatron/checkpoint_utils/llama_loader_depracated.py#L267) |
| `load_state_dict_to_megatron_llama._broadcast_tp_shard_tensor_vocab` | `def(tensor, name, chunk_dim, mutate_func)` | [`verl/models/llama/megatron/checkpoint_utils/llama_loader_depracated.py`](../../verl/models/llama/megatron/checkpoint_utils/llama_loader_depracated.py#L128) |
| `load_state_dict_to_megatron_llama._fetch_tensor` | `def(tensor, name)` | [`verl/models/llama/megatron/checkpoint_utils/llama_loader.py`](../../verl/models/llama/megatron/checkpoint_utils/llama_loader.py#L94) |
| `load_state_dict_to_megatron_llama._fetch_tp_shard_tensor` | `def(tensor, name, chunk_dim, mutate_func)` | [`verl/models/llama/megatron/checkpoint_utils/llama_loader.py`](../../verl/models/llama/megatron/checkpoint_utils/llama_loader.py#L116) |
| `load_state_dict_to_megatron_llama._fetch_tp_shard_tensor_gate_up` | `def(tensor, gate_name, up_name)` | [`verl/models/llama/megatron/checkpoint_utils/llama_loader.py`](../../verl/models/llama/megatron/checkpoint_utils/llama_loader.py#L132) |
| `load_state_dict_to_megatron_llama._fetch_tp_shard_tensor_qkv` | `def(tensor, q_name, k_name, v_name)` | [`verl/models/llama/megatron/checkpoint_utils/llama_loader.py`](../../verl/models/llama/megatron/checkpoint_utils/llama_loader.py#L154) |
| `load_state_dict_to_megatron_llama._fetch_tp_shard_tensor_vocab` | `def(tensor, name, chunk_dim, mutate_func)` | [`verl/models/llama/megatron/checkpoint_utils/llama_loader.py`](../../verl/models/llama/megatron/checkpoint_utils/llama_loader.py#L100) |
| `load_state_dict_to_megatron_llama._get_gpt_model` | `def(model)` | [`verl/models/llama/megatron/checkpoint_utils/llama_loader.py`](../../verl/models/llama/megatron/checkpoint_utils/llama_loader.py#L62) |
| `load_state_dict_to_megatron_llama._get_gpt_model` | `def(model)` | [`verl/models/llama/megatron/checkpoint_utils/llama_loader_depracated.py`](../../verl/models/llama/megatron/checkpoint_utils/llama_loader_depracated.py#L62) |
| `load_state_dict_to_megatron_llama.broadcast_params` | `def(module)` | [`verl/models/llama/megatron/checkpoint_utils/llama_loader_depracated.py`](../../verl/models/llama/megatron/checkpoint_utils/llama_loader_depracated.py#L65) |
| `load_state_dict_to_megatron_llama.fetch_params` | `def(module)` | [`verl/models/llama/megatron/checkpoint_utils/llama_loader.py`](../../verl/models/llama/megatron/checkpoint_utils/llama_loader.py#L65) |
| `load_state_dict_to_megatron_qwen2` | `def(state_dict, wrapped_models, config, params_dtype, is_value_model, tie_word_embeddings)` | [`verl/models/qwen2/megatron/checkpoint_utils/qwen2_loader.py`](../../verl/models/qwen2/megatron/checkpoint_utils/qwen2_loader.py#L49) |
| `load_state_dict_to_megatron_qwen2` | `def(state_dict, wrapped_models, config, params_dtype, is_value_model, tie_word_embeddings)` | [`verl/models/qwen2/megatron/checkpoint_utils/qwen2_loader_depracated.py`](../../verl/models/qwen2/megatron/checkpoint_utils/qwen2_loader_depracated.py#L49) |
| `load_state_dict_to_megatron_qwen2._broadcast_tensor` | `def(tensor, name)` | [`verl/models/qwen2/megatron/checkpoint_utils/qwen2_loader_depracated.py`](../../verl/models/qwen2/megatron/checkpoint_utils/qwen2_loader_depracated.py#L92) |
| `load_state_dict_to_megatron_qwen2._broadcast_tp_shard_tensor` | `def(tensor, name, chunk_dim, mutate_func)` | [`verl/models/qwen2/megatron/checkpoint_utils/qwen2_loader_depracated.py`](../../verl/models/qwen2/megatron/checkpoint_utils/qwen2_loader_depracated.py#L172) |
| `load_state_dict_to_megatron_qwen2._broadcast_tp_shard_tensor_gate_up` | `def(tensor, gate_name, up_name)` | [`verl/models/qwen2/megatron/checkpoint_utils/qwen2_loader_depracated.py`](../../verl/models/qwen2/megatron/checkpoint_utils/qwen2_loader_depracated.py#L217) |
| `load_state_dict_to_megatron_qwen2._broadcast_tp_shard_tensor_qkv` | `def(tensor, q_name, k_name, v_name, bias)` | [`verl/models/qwen2/megatron/checkpoint_utils/qwen2_loader_depracated.py`](../../verl/models/qwen2/megatron/checkpoint_utils/qwen2_loader_depracated.py#L265) |
| `load_state_dict_to_megatron_qwen2._broadcast_tp_shard_tensor_vocab` | `def(tensor, name, chunk_dim, mutate_func)` | [`verl/models/qwen2/megatron/checkpoint_utils/qwen2_loader_depracated.py`](../../verl/models/qwen2/megatron/checkpoint_utils/qwen2_loader_depracated.py#L126) |
| `load_state_dict_to_megatron_qwen2._fetch_tensor` | `def(tensor, name)` | [`verl/models/qwen2/megatron/checkpoint_utils/qwen2_loader.py`](../../verl/models/qwen2/megatron/checkpoint_utils/qwen2_loader.py#L92) |
| `load_state_dict_to_megatron_qwen2._fetch_tp_shard_tensor` | `def(tensor, name, chunk_dim, mutate_func)` | [`verl/models/qwen2/megatron/checkpoint_utils/qwen2_loader.py`](../../verl/models/qwen2/megatron/checkpoint_utils/qwen2_loader.py#L114) |
| `load_state_dict_to_megatron_qwen2._fetch_tp_shard_tensor_gate_up` | `def(tensor, gate_name, up_name)` | [`verl/models/qwen2/megatron/checkpoint_utils/qwen2_loader.py`](../../verl/models/qwen2/megatron/checkpoint_utils/qwen2_loader.py#L130) |
| `load_state_dict_to_megatron_qwen2._fetch_tp_shard_tensor_qkv` | `def(tensor, q_name, k_name, v_name, bias)` | [`verl/models/qwen2/megatron/checkpoint_utils/qwen2_loader.py`](../../verl/models/qwen2/megatron/checkpoint_utils/qwen2_loader.py#L152) |
| `load_state_dict_to_megatron_qwen2._fetch_tp_shard_tensor_vocab` | `def(tensor, name, chunk_dim, mutate_func)` | [`verl/models/qwen2/megatron/checkpoint_utils/qwen2_loader.py`](../../verl/models/qwen2/megatron/checkpoint_utils/qwen2_loader.py#L98) |
| `load_state_dict_to_megatron_qwen2._get_gpt_model` | `def(model)` | [`verl/models/qwen2/megatron/checkpoint_utils/qwen2_loader.py`](../../verl/models/qwen2/megatron/checkpoint_utils/qwen2_loader.py#L60) |
| `load_state_dict_to_megatron_qwen2._get_gpt_model` | `def(model)` | [`verl/models/qwen2/megatron/checkpoint_utils/qwen2_loader_depracated.py`](../../verl/models/qwen2/megatron/checkpoint_utils/qwen2_loader_depracated.py#L60) |
| `load_state_dict_to_megatron_qwen2.broadcast_params` | `def(module)` | [`verl/models/qwen2/megatron/checkpoint_utils/qwen2_loader_depracated.py`](../../verl/models/qwen2/megatron/checkpoint_utils/qwen2_loader_depracated.py#L63) |
| `load_state_dict_to_megatron_qwen2.fetch_params` | `def(module)` | [`verl/models/qwen2/megatron/checkpoint_utils/qwen2_loader.py`](../../verl/models/qwen2/megatron/checkpoint_utils/qwen2_loader.py#L63) |
| `load_tokenizer_and_model` | `def(local_model_path, dtype)` | [`tests/workers/rollout/utils_sglang.py`](../../tests/workers/rollout/utils_sglang.py#L88) |
| `LocalLogger.__init__` | `def(self, remote_logger, enable_wandb, print_to_console)` | [`verl/utils/logger/aggregate_logger.py`](../../verl/utils/logger/aggregate_logger.py#L33) |
| `LocalLogger.flush` | `def(self)` | [`verl/utils/logger/aggregate_logger.py`](../../verl/utils/logger/aggregate_logger.py#L38) |
| `LocalLogger.log` | `def(self, data, step)` | [`verl/utils/logger/aggregate_logger.py`](../../verl/utils/logger/aggregate_logger.py#L41) |
| `log` | `def(*args)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L255) |
| `log_gpu_memory_usage` | `def(head, logger, level, rank)` | [`verl/utils/debug/performance.py`](../../verl/utils/debug/performance.py#L49) |
| `log_print` | `def(ctn)` | [`verl/utils/debug/performance.py`](../../verl/utils/debug/performance.py#L98) |
| `log_probs_from_logits_all_rmpad` | `def(input_ids_rmpad, logits_rmpad, indices, batch_size, seqlen, response_length)` | [`verl/utils/torch_functional.py`](../../verl/utils/torch_functional.py#L419) |
| `log_probs_from_logits_response` | `def(input_ids, logits, response_length)` | [`verl/utils/torch_functional.py`](../../verl/utils/torch_functional.py#L378) |
| `log_probs_from_logits_response_rmpad` | `def(input_ids, attention_mask, logits_rmpad, response_length)` | [`verl/utils/torch_functional.py`](../../verl/utils/torch_functional.py#L394) |
| `log_seqlen_unbalance` | `def(seqlen_list, partitions, prefix)` | [`verl/utils/seqlen_balancing.py`](../../verl/utils/seqlen_balancing.py#L187) |
| `log_to_file` | `def(string)` | [`verl/utils/logging_utils.py`](../../verl/utils/logging_utils.py#L28) |
| `Logger.__init__` | `def(self, dir, output_formats)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L340) |
| `Logger._do_log` | `def(self, args)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L386) |
| `Logger.close` | `def(self)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L380) |
| `Logger.dumpkvs` | `def(self)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L360) |
| `Logger.get_dir` | `def(self)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L377) |
| `Logger.log` | `def(self, *args)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L368) |
| `Logger.logkv` | `def(self, key, val)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L349) |
| `Logger.logkv_mean` | `def(self, key, val)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L352) |
| `Logger.set_level` | `def(self, level)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L374) |
| `logkv` | `def(key, val)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L217) |
| `logkv_mean` | `def(key, val)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L226) |
| `logkvs` | `def(d)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L233) |
| `logprobs_from_logits` | `def(logits, labels, inplace_backward)` | [`verl/utils/torch_functional.py`](../../verl/utils/torch_functional.py#L54) |
| `logprobs_from_logits_flash_attn` | `def(logits, labels, inplace_backward)` | [`verl/utils/torch_functional.py`](../../verl/utils/torch_functional.py#L83) |
| `logprobs_from_logits_naive` | `def(logits, labels)` | [`verl/utils/torch_functional.py`](../../verl/utils/torch_functional.py#L89) |
| `logprobs_from_logits_v2` | `def(logits, labels)` | [`verl/utils/torch_functional.py`](../../verl/utils/torch_functional.py#L95) |
| `main` | `def()` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/train_choice_il.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/train_choice_il.py#L360) |
| `main` | `def()` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/train_rl.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/train_rl.py#L228) |
| `main` | `def()` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/attributes/annotate.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/attributes/annotate.py#L62) |
| `main` | `def()` | [`examples/data_preprocess/multiturn.py`](../../examples/data_preprocess/multiturn.py#L24) |
| `main` | `def()` | [`examples/data_preprocess/prepare_sdar_multitask.py`](../../examples/data_preprocess/prepare_sdar_multitask.py#L127) |
| `main` | `def()` | [`examples/data_preprocess/preprocess_search_r1_dataset.py`](../../examples/data_preprocess/preprocess_search_r1_dataset.py#L95) |
| `main` | `def(config)` | [`examples/split_placement/main_ppo_split.py`](../../examples/split_placement/main_ppo_split.py#L94) |
| `main` | `def(config)` | [`recipe/dapo/main_dapo.py`](../../recipe/dapo/main_dapo.py#L30) |
| `main` | `def(config)` | [`recipe/hgpo/main_hgpo.py`](../../recipe/hgpo/main_hgpo.py#L29) |
| `main` | `def(config)` | [`recipe/prime/main_prime.py`](../../recipe/prime/main_prime.py#L39) |
| `main` | `def(config)` | [`recipe/r1/main_eval.py`](../../recipe/r1/main_eval.py#L41) |
| `main` | `def(config)` | [`recipe/spin/main_spin.py`](../../recipe/spin/main_spin.py#L26) |
| `main` | `def(config)` | [`recipe/sppo/main_sppo.py`](../../recipe/sppo/main_sppo.py#L32) |
| `main` | `def()` | [`scripts/model_merger.py`](../../scripts/model_merger.py#L560) |
| `main` | `def(config)` | [`tests/e2e/arithmetic_sequence/rl/main_trainer.py`](../../tests/e2e/arithmetic_sequence/rl/main_trainer.py#L84) |
| `main` | `def(config)` | [`tests/e2e/sft/test_sp_loss_match.py`](../../tests/e2e/sft/test_sp_loss_match.py#L119) |
| `main` | `def()` | [`tests/workers/rollout/run_fsdp_vllm.py`](../../tests/workers/rollout/run_fsdp_vllm.py#L30) |
| `main` | `def(config)` | [`verl/trainer/fsdp_sft_trainer.py`](../../verl/trainer/fsdp_sft_trainer.py#L566) |
| `main` | `def(config)` | [`verl/trainer/main_eval.py`](../../verl/trainer/main_eval.py#L40) |
| `main` | `def(config)` | [`verl/trainer/main_generation.py`](../../verl/trainer/main_generation.py#L45) |
| `main` | `def(config)` | [`verl/trainer/main_opd.py`](../../verl/trainer/main_opd.py#L16) |
| `main` | `def(config)` | [`verl/trainer/main_ppo.py`](../../verl/trainer/main_ppo.py#L30) |
| `main` | `def(config)` | [`verl/trainer/main_rlsd.py`](../../verl/trainer/main_rlsd.py#L12) |
| `main` | `def(config)` | [`verl/trainer/main_sdar.py`](../../verl/trainer/main_sdar.py#L13) |
| `main` | `def(config)` | [`verl/trainer/main_skill_grpo.py`](../../verl/trainer/main_skill_grpo.py#L15) |
| `main` | `def(config)` | [`verl/trainer/main_skillsd.py`](../../verl/trainer/main_skillsd.py#L12) |
| `main.apply_process_row` | `def(row, split_name)` | [`examples/data_preprocess/preprocess_search_r1_dataset.py`](../../examples/data_preprocess/preprocess_search_r1_dataset.py#L122) |
| `main_task` | `def(config)` | [`examples/split_placement/main_ppo_split.py`](../../examples/split_placement/main_ppo_split.py#L106) |
| `main_task` | `def(config, compute_score)` | [`recipe/prime/main_prime.py`](../../recipe/prime/main_prime.py#L55) |
| `main_task` | `def(config)` | [`verl/trainer/main_generation.py`](../../verl/trainer/main_generation.py#L61) |
| `make_batch_generator` | `def(batches, vpp_size)` | [`verl/utils/megatron/pipeline_parallel.py`](../../verl/utils/megatron/pipeline_parallel.py#L49) |
| `make_envs` | `def(config)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L956) |
| `make_envs` | `def(config)` | [`recipe/hgpo/env_manager.py`](../../recipe/hgpo/env_manager.py#L591) |
| `make_map_fn` | `def(split)` | [`examples/data_preprocess/aime2024_multiturn_w_tool.py`](../../examples/data_preprocess/aime2024_multiturn_w_tool.py#L40) |
| `make_map_fn` | `def(split)` | [`examples/data_preprocess/dapo_multiturn_w_tool.py`](../../examples/data_preprocess/dapo_multiturn_w_tool.py#L40) |
| `make_map_fn` | `def(split)` | [`examples/data_preprocess/geo3k.py`](../../examples/data_preprocess/geo3k.py#L45) |
| `make_map_fn` | `def(split)` | [`examples/data_preprocess/gsm8k.py`](../../examples/data_preprocess/gsm8k.py#L52) |
| `make_map_fn` | `def(split)` | [`examples/data_preprocess/gsm8k_multiturn_w_tool.py`](../../examples/data_preprocess/gsm8k_multiturn_w_tool.py#L53) |
| `make_map_fn` | `def(split)` | [`examples/data_preprocess/hellaswag.py`](../../examples/data_preprocess/hellaswag.py#L54) |
| `make_map_fn` | `def(split)` | [`examples/data_preprocess/math_dataset.py`](../../examples/data_preprocess/math_dataset.py#L50) |
| `make_map_fn` | `def(split)` | [`examples/data_preprocess/prepare.py`](../../examples/data_preprocess/prepare.py#L54) |
| `make_map_fn.process_fn` | `def(example, idx)` | [`examples/data_preprocess/aime2024_multiturn_w_tool.py`](../../examples/data_preprocess/aime2024_multiturn_w_tool.py#L41) |
| `make_map_fn.process_fn` | `def(example, idx)` | [`examples/data_preprocess/dapo_multiturn_w_tool.py`](../../examples/data_preprocess/dapo_multiturn_w_tool.py#L41) |
| `make_map_fn.process_fn` | `def(example, idx)` | [`examples/data_preprocess/geo3k.py`](../../examples/data_preprocess/geo3k.py#L46) |
| `make_map_fn.process_fn` | `def(example, idx)` | [`examples/data_preprocess/gsm8k.py`](../../examples/data_preprocess/gsm8k.py#L53) |
| `make_map_fn.process_fn` | `def(example, idx)` | [`examples/data_preprocess/gsm8k_multiturn_w_tool.py`](../../examples/data_preprocess/gsm8k_multiturn_w_tool.py#L54) |
| `make_map_fn.process_fn` | `def(doc, idx)` | [`examples/data_preprocess/hellaswag.py`](../../examples/data_preprocess/hellaswag.py#L55) |
| `make_map_fn.process_fn` | `def(example, idx)` | [`examples/data_preprocess/math_dataset.py`](../../examples/data_preprocess/math_dataset.py#L51) |
| `make_map_fn.process_fn` | `def(example, idx)` | [`examples/data_preprocess/prepare.py`](../../examples/data_preprocess/prepare.py#L56) |
| `make_mock_env` | `def(done_at)` | [`tests/ray_cpu/test_async_rollout_equivalence.py`](../../tests/ray_cpu/test_async_rollout_equivalence.py#L60) |
| `make_mock_env.env_step_sync` | `def(traj_id, step, action)` | [`tests/ray_cpu/test_async_rollout_equivalence.py`](../../tests/ray_cpu/test_async_rollout_equivalence.py#L63) |
| `make_output_format` | `def(format, ev_dir, log_suffix, args)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L195) |
| `make_reward_function` | `def(tokenizer, num_examine)` | [`tests/e2e/arithmetic_sequence/rl/main_trainer.py`](../../tests/e2e/arithmetic_sequence/rl/main_trainer.py#L30) |
| `make_reward_function.arithmetic_sequence_reward_function` | `def(data, return_dict)` | [`tests/e2e/arithmetic_sequence/rl/main_trainer.py`](../../tests/e2e/arithmetic_sequence/rl/main_trainer.py#L31) |
| `makedirs` | `def(name, mode, exist_ok, **kwargs)` | [`verl/utils/hdfs_io.py`](../../verl/utils/hdfs_io.py#L50) |
| `map_action_to_html` | `def(action, **kwargs)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/engine.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/engine.py#L44) |
| `mapping_string_to_attn_backend` | `def(args)` | [`verl/models/mcore/config_converter.py`](../../verl/models/mcore/config_converter.py#L391) |
| `mark_parameter_as_sequence_parallel` | `def(parameter)` | [`verl/utils/megatron/sequence_parallel.py`](../../verl/utils/megatron/sequence_parallel.py#L21) |
| `masked_mean` | `def(values, mask, axis)` | [`verl/utils/torch_functional.py`](../../verl/utils/torch_functional.py#L136) |
| `masked_sum` | `def(values, mask, axis)` | [`verl/utils/torch_functional.py`](../../verl/utils/torch_functional.py#L131) |
| `masked_var` | `def(values, mask, unbiased)` | [`verl/utils/torch_functional.py`](../../verl/utils/torch_functional.py#L152) |
| `masked_whiten` | `def(values, mask, shift_mean)` | [`verl/utils/torch_functional.py`](../../verl/utils/torch_functional.py#L170) |
| `match_answer` | `def(response)` | [`verl/utils/reward_score/prime_math/__init__.py`](../../verl/utils/reward_score/prime_math/__init__.py#L337) |
| `math_equal` | `def(prediction, reference, include_percentage, tolerance, timeout, pi)` | [`verl/utils/reward_score/prime_math/grader.py`](../../verl/utils/reward_score/prime_math/grader.py#L173) |
| `mcore_model_parallel_config` | `def(sequence_parallel, params_dtype)` | [`verl/utils/megatron_utils.py`](../../verl/utils/megatron_utils.py#L214) |
| `McoreToHFWeightConverterBase.__init__` | `def(self, hf_config, mcore_config)` | [`verl/models/mcore/weight_converter.py`](../../verl/models/mcore/weight_converter.py#L26) |
| `McoreToHFWeightConverterBase.convert_param` | `def(self, name, params_one_group)` | [`verl/models/mcore/weight_converter.py`](../../verl/models/mcore/weight_converter.py#L30) |
| `McoreToHFWeightConverterDense._convert_attention_param` | `def(self, name, params)` | [`verl/models/mcore/weight_converter.py`](../../verl/models/mcore/weight_converter.py#L35) |
| `McoreToHFWeightConverterDense._convert_mlp_param` | `def(self, name, params)` | [`verl/models/mcore/weight_converter.py`](../../verl/models/mcore/weight_converter.py#L65) |
| `McoreToHFWeightConverterDense.convert_param` | `def(self, name, params_one_group)` | [`verl/models/mcore/weight_converter.py`](../../verl/models/mcore/weight_converter.py#L86) |
| `McoreToHFWeightConverterDpskv3._convert_attention_param` | `def(self, name, params)` | [`verl/models/mcore/weight_converter.py`](../../verl/models/mcore/weight_converter.py#L151) |
| `McoreToHFWeightConverterDpskv3._convert_mlp_param` | `def(self, name, params)` | [`verl/models/mcore/weight_converter.py`](../../verl/models/mcore/weight_converter.py#L190) |
| `McoreToHFWeightConverterDpskv3.convert_param` | `def(self, name, params_one_group)` | [`verl/models/mcore/weight_converter.py`](../../verl/models/mcore/weight_converter.py#L260) |
| `McoreToHFWeightConverterMixtral._convert_mlp_param` | `def(self, name, params)` | [`verl/models/mcore/weight_converter.py`](../../verl/models/mcore/weight_converter.py#L278) |
| `McoreToHFWeightConverterQwen2Moe._convert_mlp_param` | `def(self, name, params)` | [`verl/models/mcore/weight_converter.py`](../../verl/models/mcore/weight_converter.py#L104) |
| `McoreToHFWeightConverterQwen3Moe._convert_mlp_param` | `def(self, name, params)` | [`verl/models/mcore/weight_converter.py`](../../verl/models/mcore/weight_converter.py#L302) |
| `md5_encode` | `def(path)` | [`verl/utils/fs.py`](../../verl/utils/fs.py#L47) |
| `mean_util_between` | `def(t0, t1)` | [`verl/utils/gpu_profiler.py`](../../verl/utils/gpu_profiler.py#L343) |
| `MegatronCheckpointManager.__init__` | `def(self, config, model_config, role, model, arch, hf_config, param_dtype, share_embeddings_and_output_weights, tokenizer, optimizer, use_distributed_optimizer, checkpoint_contents, **kwargs)` | [`verl/utils/checkpoint/megatron_checkpoint_manager.py`](../../verl/utils/checkpoint/megatron_checkpoint_manager.py#L54) |
| `MegatronCheckpointManager.get_checkpoint_name` | `def(self, checkpoints_path, pipeline_parallel, tensor_rank, pipeline_rank, cp_rank, expert_parallel, expert_rank, return_base_dir, basename)` | [`verl/utils/checkpoint/megatron_checkpoint_manager.py`](../../verl/utils/checkpoint/megatron_checkpoint_manager.py#L130) |
| `MegatronCheckpointManager.get_rng_state` | `def(self, use_dist_ckpt, data_parallel_random_init)` | [`verl/utils/checkpoint/megatron_checkpoint_manager.py`](../../verl/utils/checkpoint/megatron_checkpoint_manager.py#L96) |
| `MegatronCheckpointManager.load_checkpoint` | `def(self, local_path, hdfs_path, del_local_after_load)` | [`verl/utils/checkpoint/megatron_checkpoint_manager.py`](../../verl/utils/checkpoint/megatron_checkpoint_manager.py#L198) |
| `MegatronCheckpointManager.load_optimizer` | `def(self, ckpt_path)` | [`verl/utils/checkpoint/megatron_checkpoint_manager.py`](../../verl/utils/checkpoint/megatron_checkpoint_manager.py#L176) |
| `MegatronCheckpointManager.load_rng_states` | `def(self, ckpt_path, data_parallel_random_init, use_dist_ckpt)` | [`verl/utils/checkpoint/megatron_checkpoint_manager.py`](../../verl/utils/checkpoint/megatron_checkpoint_manager.py#L182) |
| `MegatronCheckpointManager.save_checkpoint` | `def(self, local_path, hdfs_path, global_step, max_ckpt_to_keep)` | [`verl/utils/checkpoint/megatron_checkpoint_manager.py`](../../verl/utils/checkpoint/megatron_checkpoint_manager.py#L223) |
| `MegatronConfig.__init__` | `def(self)` | [`scripts/converter_hf_to_mcore.py`](../../scripts/converter_hf_to_mcore.py#L44) |
| `MegatronMemoryBufferForRollout.__init__` | `def(self, transform_memory_param_fn)` | [`verl/utils/memory_buffer.py`](../../verl/utils/memory_buffer.py#L175) |
| `MegatronMemoryBufferForRollout.build_memory_reference` | `def(self)` | [`verl/utils/memory_buffer.py`](../../verl/utils/memory_buffer.py#L199) |
| `MegatronMemoryBufferForRollout.initialize_weight_buffer` | `def(self, weight_buffer_meta_pp)` | [`verl/utils/memory_buffer.py`](../../verl/utils/memory_buffer.py#L181) |
| `MegatronMemoryBufferForRollout.memory_buffers` | `def(self)` | [`verl/utils/memory_buffer.py`](../../verl/utils/memory_buffer.py#L213) |
| `MegatronMemoryBufferForRollout.named_parameters` | `def(self)` | [`verl/utils/memory_buffer.py`](../../verl/utils/memory_buffer.py#L205) |
| `MegatronMemoryBufferForRollout.weight_buffers` | `def(self)` | [`verl/utils/memory_buffer.py`](../../verl/utils/memory_buffer.py#L209) |
| `MegatronModelMerger.__init__` | `def(self, config)` | [`scripts/model_merger.py`](../../scripts/model_merger.py#L336) |
| `MegatronModelMerger._check_megatron_checkpoint_path` | `def(self, model_path)` | [`scripts/model_merger.py`](../../scripts/model_merger.py#L349) |
| `MegatronModelMerger._get_tp_pp_rank_from_sharded_dir` | `def(self, sharded_dir)` | [`scripts/model_merger.py`](../../scripts/model_merger.py#L342) |
| `MegatronModelMerger._load_state_dicts` | `def(self, model_ckpt_path, sharded_dirs, tp_size, pp_size)` | [`scripts/model_merger.py`](../../scripts/model_merger.py#L415) |
| `MegatronModelMerger._load_state_dicts._process_one_megatron_shard` | `def(sharded_dir)` | [`scripts/model_merger.py`](../../scripts/model_merger.py#L418) |
| `MegatronModelMerger._merge_across_tp` | `def(self, key, tp_data, config, tp_size, is_value_model)` | [`scripts/model_merger.py`](../../scripts/model_merger.py#L364) |
| `MegatronModelMerger._merge_state_dicts` | `def(self, model_state_dict_lst, tp_size, pp_size)` | [`scripts/model_merger.py`](../../scripts/model_merger.py#L431) |
| `MegatronModelMerger._replace_name` | `def(self, megatron_name, name_mapping)` | [`scripts/model_merger.py`](../../scripts/model_merger.py#L536) |
| `MegatronModelMerger._test_state_dict` | `def(self, state_dict)` | [`scripts/model_merger.py`](../../scripts/model_merger.py#L496) |
| `MegatronModelMerger.merge_and_save` | `def(self)` | [`scripts/model_merger.py`](../../scripts/model_merger.py#L474) |
| `MegatronPPOActor.__init__` | `def(self, config, model_config, hf_config, tf_config, actor_module, actor_optimizer)` | [`verl/workers/actor/megatron_actor.py`](../../verl/workers/actor/megatron_actor.py#L59) |
| `MegatronPPOActor._validate_config` | `def(self, config)` | [`verl/workers/actor/megatron_actor.py`](../../verl/workers/actor/megatron_actor.py#L140) |
| `MegatronPPOActor.compute_log_prob` | `def(self, data, calculate_entropy)` | [`verl/workers/actor/megatron_actor.py`](../../verl/workers/actor/megatron_actor.py#L151) |
| `MegatronPPOActor.compute_log_prob.compute_logprobs_fn` | `def(output, data, use_dynamic_bsz, indices)` | [`verl/workers/actor/megatron_actor.py`](../../verl/workers/actor/megatron_actor.py#L178) |
| `MegatronPPOActor.forward_backward_batch` | `def(self, data, forward_only, post_process_fn, calculate_entropy, use_dynamic_bsz, micro_batch_size, max_token_len, mini_batch_size)` | [`verl/workers/actor/megatron_actor.py`](../../verl/workers/actor/megatron_actor.py#L282) |
| `MegatronPPOActor.forward_backward_batch.forward_step` | `def(batch_iter, model)` | [`verl/workers/actor/megatron_actor.py`](../../verl/workers/actor/megatron_actor.py#L406) |
| `MegatronPPOActor.forward_backward_batch.forward_step.logits_processor` | `def(logits, label, label_mask)` | [`verl/workers/actor/megatron_actor.py`](../../verl/workers/actor/megatron_actor.py#L420) |
| `MegatronPPOActor.forward_backward_batch.loss_func` | `def(output, data, meta_info)` | [`verl/workers/actor/megatron_actor.py`](../../verl/workers/actor/megatron_actor.py#L316) |
| `MegatronPPOActor.make_minibatch_iterator` | `def(self, data)` | [`verl/workers/actor/megatron_actor.py`](../../verl/workers/actor/megatron_actor.py#L247) |
| `MegatronPPOActor.update_policy` | `def(self, dataloader)` | [`verl/workers/actor/megatron_actor.py`](../../verl/workers/actor/megatron_actor.py#L486) |
| `MegatronPPOCritic.__init__` | `def(self, config, model_config, hf_config, tf_config, critic_module, critic_optimizer, critic_optimizer_config)` | [`verl/workers/critic/megatron_critic.py`](../../verl/workers/critic/megatron_critic.py#L46) |
| `MegatronPPOCritic._validate_config` | `def(self, config)` | [`verl/workers/critic/megatron_critic.py`](../../verl/workers/critic/megatron_critic.py#L81) |
| `MegatronPPOCritic.compute_values` | `def(self, data)` | [`verl/workers/critic/megatron_critic.py`](../../verl/workers/critic/megatron_critic.py#L92) |
| `MegatronPPOCritic.forward_backward_batch` | `def(self, data, forward_only, use_dynamic_bsz, micro_batch_size, max_token_len, mini_batch_size)` | [`verl/workers/critic/megatron_critic.py`](../../verl/workers/critic/megatron_critic.py#L146) |
| `MegatronPPOCritic.forward_backward_batch.forward_step` | `def(batch_iter, model)` | [`verl/workers/critic/megatron_critic.py`](../../verl/workers/critic/megatron_critic.py#L210) |
| `MegatronPPOCritic.forward_backward_batch.loss_func` | `def(output, data, meta_info)` | [`verl/workers/critic/megatron_critic.py`](../../verl/workers/critic/megatron_critic.py#L174) |
| `MegatronPPOCritic.make_minibatch_iterator` | `def(self, data)` | [`verl/workers/critic/megatron_critic.py`](../../verl/workers/critic/megatron_critic.py#L136) |
| `MegatronPPOCritic.update_critic` | `def(self, dataloader)` | [`verl/workers/critic/megatron_critic.py`](../../verl/workers/critic/megatron_critic.py#L262) |
| `MegatronRayWorkerGroup.__init__` | `def(self, resource_pool, ray_cls_with_init, default_megatron_kwargs, **kwargs)` | [`verl/single_controller/ray/megatron.py`](../../verl/single_controller/ray/megatron.py#L52) |
| `MegatronRayWorkerGroup.init_megatron` | `def(self, default_megatron_kwargs)` | [`verl/single_controller/ray/megatron.py`](../../verl/single_controller/ray/megatron.py#L69) |
| `MegatronRewardModel.__init__` | `def(self, config, model_config, reward_model_module, hf_config, tf_config, sft_tokenizer, rm_tokenizer)` | [`verl/workers/reward_model/megatron/reward_model.py`](../../verl/workers/reward_model/megatron/reward_model.py#L34) |
| `MegatronRewardModel.compute_reward` | `def(self, data)` | [`verl/workers/reward_model/megatron/reward_model.py`](../../verl/workers/reward_model/megatron/reward_model.py#L124) |
| `MegatronRewardModel.forward_batch` | `def(self, data, use_dynamic_bsz, micro_batch_size, max_token_len)` | [`verl/workers/reward_model/megatron/reward_model.py`](../../verl/workers/reward_model/megatron/reward_model.py#L203) |
| `MegatronRewardModel.forward_batch.forward_step` | `def(batch_iter, model)` | [`verl/workers/reward_model/megatron/reward_model.py`](../../verl/workers/reward_model/megatron/reward_model.py#L241) |
| `MegatronRewardModel.forward_batch.loss_func` | `def(output)` | [`verl/workers/reward_model/megatron/reward_model.py`](../../verl/workers/reward_model/megatron/reward_model.py#L238) |
| `MegatronRewardModel.load_params_to_cuda` | `def(self)` | [`verl/workers/reward_model/megatron/reward_model.py`](../../verl/workers/reward_model/megatron/reward_model.py#L300) |
| `MegatronRewardModel.offload_params_to_cpu` | `def(self)` | [`verl/workers/reward_model/megatron/reward_model.py`](../../verl/workers/reward_model/megatron/reward_model.py#L292) |
| `MegatronRewardModel.re_encode_by_rm_tokenizer` | `def(self, data)` | [`verl/workers/reward_model/megatron/reward_model.py`](../../verl/workers/reward_model/megatron/reward_model.py#L59) |
| `MegatronSGLangShardingManager.__enter__` | `def(self)` | [`verl/workers/sharding_manager/megatron_sglang.py`](../../verl/workers/sharding_manager/megatron_sglang.py#L84) |
| `MegatronSGLangShardingManager.__exit__` | `def(self, exc_type, exc_value, traceback)` | [`verl/workers/sharding_manager/megatron_sglang.py`](../../verl/workers/sharding_manager/megatron_sglang.py#L100) |
| `MegatronSGLangShardingManager.__init__` | `def(self, actor_module, inference_engine, model_config, transformer_config, layer_name_mapping, weight_converter, device_mesh)` | [`verl/workers/sharding_manager/megatron_sglang.py`](../../verl/workers/sharding_manager/megatron_sglang.py#L49) |
| `MegatronSGLangShardingManager.postprocess_data` | `def(self, data)` | [`verl/workers/sharding_manager/megatron_sglang.py`](../../verl/workers/sharding_manager/megatron_sglang.py#L152) |
| `MegatronSGLangShardingManager.preprocess_data` | `def(self, data)` | [`verl/workers/sharding_manager/megatron_sglang.py`](../../verl/workers/sharding_manager/megatron_sglang.py#L144) |
| `MegatronSGLangShardingManager.release_memory` | `def(self)` | [`verl/workers/sharding_manager/megatron_sglang.py`](../../verl/workers/sharding_manager/megatron_sglang.py#L139) |
| `MegatronSGLangShardingManager.update_weights` | `def(self, params)` | [`verl/workers/sharding_manager/megatron_sglang.py`](../../verl/workers/sharding_manager/megatron_sglang.py#L115) |
| `MegatronVLLMShardingManager.__enter__` | `def(self)` | [`verl/workers/sharding_manager/megatron_vllm.py`](../../verl/workers/sharding_manager/megatron_vllm.py#L308) |
| `MegatronVLLMShardingManager.__exit__` | `def(self, exc_type, exc_value, traceback)` | [`verl/workers/sharding_manager/megatron_vllm.py`](../../verl/workers/sharding_manager/megatron_vllm.py#L338) |
| `MegatronVLLMShardingManager.__init__` | `def(self, actor_module, inference_engine, model_config, transformer_config, layer_name_mapping, weight_converter, module)` | [`verl/workers/sharding_manager/megatron_vllm.py`](../../verl/workers/sharding_manager/megatron_vllm.py#L268) |
| `MegatronVLLMShardingManager.postprocess_data` | `def(self, data)` | [`verl/workers/sharding_manager/megatron_vllm.py`](../../verl/workers/sharding_manager/megatron_vllm.py#L360) |
| `MegatronVLLMShardingManager.preprocess_data` | `def(self, data)` | [`verl/workers/sharding_manager/megatron_vllm.py`](../../verl/workers/sharding_manager/megatron_vllm.py#L352) |
| `MegatronWorker.__init__` | `def(self, cuda_visible_devices)` | [`verl/single_controller/base/megatron/worker.py`](../../verl/single_controller/base/megatron/worker.py#L19) |
| `MegatronWorker._init_hf_config_and_tf_config` | `def(self, model_path, dtype, override_model_config, override_transformer_config, trust_remote_code)` | [`verl/single_controller/base/megatron/worker.py`](../../verl/single_controller/base/megatron/worker.py#L42) |
| `MegatronWorker._init_hf_config_and_tf_config.add_optimization_config_to_tf_config` | `def(tf_config)` | [`verl/single_controller/base/megatron/worker.py`](../../verl/single_controller/base/megatron/worker.py#L78) |
| `MegatronWorker.get_megatron_global_info` | `def(self)` | [`verl/single_controller/base/megatron/worker.py`](../../verl/single_controller/base/megatron/worker.py#L22) |
| `MegatronWorker.get_megatron_rank_info` | `def(self)` | [`verl/single_controller/base/megatron/worker.py`](../../verl/single_controller/base/megatron/worker.py#L32) |
| `MegatronWorkerGroup.__init__` | `def(self, resource_pool, **kwargs)` | [`verl/single_controller/base/megatron/worker_group.py`](../../verl/single_controller/base/megatron/worker_group.py#L23) |
| `MegatronWorkerGroup.cp_size` | `def(self)` | [`verl/single_controller/base/megatron/worker_group.py`](../../verl/single_controller/base/megatron/worker_group.py#L51) |
| `MegatronWorkerGroup.dp_size` | `def(self)` | [`verl/single_controller/base/megatron/worker_group.py`](../../verl/single_controller/base/megatron/worker_group.py#L41) |
| `MegatronWorkerGroup.get_megatron_global_info` | `def(self)` | [`verl/single_controller/base/megatron/worker_group.py`](../../verl/single_controller/base/megatron/worker_group.py#L55) |
| `MegatronWorkerGroup.get_megatron_rank_info` | `def(self, rank)` | [`verl/single_controller/base/megatron/worker_group.py`](../../verl/single_controller/base/megatron/worker_group.py#L31) |
| `MegatronWorkerGroup.init_megatron` | `def(self, default_megatron_kwargs)` | [`verl/single_controller/base/megatron/worker_group.py`](../../verl/single_controller/base/megatron/worker_group.py#L28) |
| `MegatronWorkerGroup.pp_size` | `def(self)` | [`verl/single_controller/base/megatron/worker_group.py`](../../verl/single_controller/base/megatron/worker_group.py#L46) |
| `MegatronWorkerGroup.tp_size` | `def(self)` | [`verl/single_controller/base/megatron/worker_group.py`](../../verl/single_controller/base/megatron/worker_group.py#L36) |
| `MemoryBuffer.__init__` | `def(self, numel, numel_padded, dtype)` | [`verl/utils/megatron/memory.py`](../../verl/utils/megatron/memory.py#L19) |
| `MemoryBuffer.__init__` | `def(self, numel, numel_padded, dtype, source)` | [`verl/utils/memory_buffer.py`](../../verl/utils/memory_buffer.py#L30) |
| `MemoryBuffer.get` | `def(self, shape, start_index)` | [`verl/utils/megatron/memory.py`](../../verl/utils/megatron/memory.py#L29) |
| `MemoryBuffer.get` | `def(self, shape, start_index)` | [`verl/utils/memory_buffer.py`](../../verl/utils/memory_buffer.py#L43) |
| `MemoryBuffer.zero` | `def(self)` | [`verl/utils/megatron/memory.py`](../../verl/utils/megatron/memory.py#L25) |
| `MemoryBuffer.zero` | `def(self)` | [`verl/utils/memory_buffer.py`](../../verl/utils/memory_buffer.py#L39) |
| `MemoryBufferModuleWrapper.__init__` | `def(self, module)` | [`verl/utils/memory_buffer.py`](../../verl/utils/memory_buffer.py#L146) |
| `MemoryBufferModuleWrapper.get_memory_buffers` | `def(self)` | [`verl/utils/memory_buffer.py`](../../verl/utils/memory_buffer.py#L153) |
| `MemoryBufferModuleWrapper.get_weight_buffer_meta` | `def(self)` | [`verl/utils/memory_buffer.py`](../../verl/utils/memory_buffer.py#L156) |
| `merge_megatron_ckpt_gptmodel` | `def(wrapped_models, config, dtype, is_value_model, tie_word_embeddings)` | [`verl/models/mcore/saver.py`](../../verl/models/mcore/saver.py#L75) |
| `merge_megatron_ckpt_gptmodel._broadcast_tensor` | `def(tensor, name, src_pp_rank)` | [`verl/models/mcore/saver.py`](../../verl/models/mcore/saver.py#L129) |
| `merge_megatron_ckpt_gptmodel._broadcast_tp_shard_tensor` | `def(tensor, name, src_pp_rank, concat_dim, mutate_func)` | [`verl/models/mcore/saver.py`](../../verl/models/mcore/saver.py#L168) |
| `merge_megatron_ckpt_gptmodel._broadcast_tp_shard_tensor_gate_up` | `def(tensor, gate_name, up_name, src_pp_rank)` | [`verl/models/mcore/saver.py`](../../verl/models/mcore/saver.py#L209) |
| `merge_megatron_ckpt_gptmodel._broadcast_tp_shard_tensor_qkv` | `def(tensor, q_name, k_name, v_name, src_pp_rank)` | [`verl/models/mcore/saver.py`](../../verl/models/mcore/saver.py#L259) |
| `merge_megatron_ckpt_gptmodel._get_cpu_tensor` | `def(tensor)` | [`verl/models/mcore/saver.py`](../../verl/models/mcore/saver.py#L122) |
| `merge_megatron_ckpt_gptmodel._get_gpt_model` | `def(model)` | [`verl/models/mcore/saver.py`](../../verl/models/mcore/saver.py#L92) |
| `merge_megatron_ckpt_gptmodel_dpskv3` | `def(wrapped_models, config, dtype, is_value_model, tie_word_embeddings)` | [`verl/models/mcore/saver.py`](../../verl/models/mcore/saver.py#L470) |
| `merge_megatron_ckpt_gptmodel_mixtral` | `def(wrapped_models, config, dtype, is_value_model, tie_word_embeddings)` | [`verl/models/mcore/saver.py`](../../verl/models/mcore/saver.py#L474) |
| `merge_megatron_ckpt_gptmodel_qwen_moe` | `def(wrapped_models, config, dtype, is_value_model, tie_word_embeddings)` | [`verl/models/mcore/saver.py`](../../verl/models/mcore/saver.py#L466) |
| `merge_megatron_ckpt_llama` | `def(wrapped_models, config, dtype, is_value_model, tie_word_embeddings)` | [`verl/models/llama/megatron/checkpoint_utils/llama_saver.py`](../../verl/models/llama/megatron/checkpoint_utils/llama_saver.py#L66) |
| `merge_megatron_ckpt_llama._broadcast_tensor` | `def(tensor, name, src_pp_rank)` | [`verl/models/llama/megatron/checkpoint_utils/llama_saver.py`](../../verl/models/llama/megatron/checkpoint_utils/llama_saver.py#L119) |
| `merge_megatron_ckpt_llama._broadcast_tp_shard_tensor` | `def(tensor, name, src_pp_rank, concat_dim, mutate_func)` | [`verl/models/llama/megatron/checkpoint_utils/llama_saver.py`](../../verl/models/llama/megatron/checkpoint_utils/llama_saver.py#L158) |
| `merge_megatron_ckpt_llama._broadcast_tp_shard_tensor_gate_up` | `def(tensor, gate_name, up_name, src_pp_rank)` | [`verl/models/llama/megatron/checkpoint_utils/llama_saver.py`](../../verl/models/llama/megatron/checkpoint_utils/llama_saver.py#L198) |
| `merge_megatron_ckpt_llama._broadcast_tp_shard_tensor_qkv` | `def(tensor, q_name, k_name, v_name, src_pp_rank)` | [`verl/models/llama/megatron/checkpoint_utils/llama_saver.py`](../../verl/models/llama/megatron/checkpoint_utils/llama_saver.py#L247) |
| `merge_megatron_ckpt_llama._get_cpu_tensor` | `def(tensor)` | [`verl/models/llama/megatron/checkpoint_utils/llama_saver.py`](../../verl/models/llama/megatron/checkpoint_utils/llama_saver.py#L112) |
| `merge_megatron_ckpt_llama._get_gpt_model` | `def(model)` | [`verl/models/llama/megatron/checkpoint_utils/llama_saver.py`](../../verl/models/llama/megatron/checkpoint_utils/llama_saver.py#L83) |
| `merge_megatron_ckpt_qwen2` | `def(wrapped_models, config, dtype, is_value_model, tie_word_embeddings)` | [`verl/models/qwen2/megatron/checkpoint_utils/qwen2_saver.py`](../../verl/models/qwen2/megatron/checkpoint_utils/qwen2_saver.py#L66) |
| `merge_megatron_ckpt_qwen2._broadcast_tensor` | `def(tensor, name, src_pp_rank)` | [`verl/models/qwen2/megatron/checkpoint_utils/qwen2_saver.py`](../../verl/models/qwen2/megatron/checkpoint_utils/qwen2_saver.py#L119) |
| `merge_megatron_ckpt_qwen2._broadcast_tp_shard_tensor` | `def(tensor, name, src_pp_rank, concat_dim, mutate_func)` | [`verl/models/qwen2/megatron/checkpoint_utils/qwen2_saver.py`](../../verl/models/qwen2/megatron/checkpoint_utils/qwen2_saver.py#L158) |
| `merge_megatron_ckpt_qwen2._broadcast_tp_shard_tensor_gate_up` | `def(tensor, gate_name, up_name, src_pp_rank)` | [`verl/models/qwen2/megatron/checkpoint_utils/qwen2_saver.py`](../../verl/models/qwen2/megatron/checkpoint_utils/qwen2_saver.py#L198) |
| `merge_megatron_ckpt_qwen2._broadcast_tp_shard_tensor_qkv` | `def(tensor, q_name, k_name, v_name, src_pp_rank)` | [`verl/models/qwen2/megatron/checkpoint_utils/qwen2_saver.py`](../../verl/models/qwen2/megatron/checkpoint_utils/qwen2_saver.py#L247) |
| `merge_megatron_ckpt_qwen2._get_cpu_tensor` | `def(tensor)` | [`verl/models/qwen2/megatron/checkpoint_utils/qwen2_saver.py`](../../verl/models/qwen2/megatron/checkpoint_utils/qwen2_saver.py#L112) |
| `merge_megatron_ckpt_qwen2._get_gpt_model` | `def(model)` | [`verl/models/qwen2/megatron/checkpoint_utils/qwen2_saver.py`](../../verl/models/qwen2/megatron/checkpoint_utils/qwen2_saver.py#L83) |
| `merge_resource_pool` | `def(rp1, rp2)` | [`verl/single_controller/ray/base.py`](../../verl/single_controller/ray/base.py#L148) |
| `MergedColumnParallelLinear.__init__` | `def(self, input_size, gate_ouput_size, up_output_size, **kwargs)` | [`verl/models/llama/megatron/layers/parallel_linear.py`](../../verl/models/llama/megatron/layers/parallel_linear.py#L55) |
| `MergedColumnParallelLinear.__init__` | `def(self, input_size, gate_ouput_size, up_output_size, **kwargs)` | [`verl/models/qwen2/megatron/layers/parallel_linear.py`](../../verl/models/qwen2/megatron/layers/parallel_linear.py#L55) |
| `meta_device_init` | `def()` | [`verl/utils/fsdp_utils.py`](../../verl/utils/fsdp_utils.py#L212) |
| `meta_device_init.register_empty_parameter` | `def(module, name, param)` | [`verl/utils/fsdp_utils.py`](../../verl/utils/fsdp_utils.py#L224) |
| `MixtralModel.get_transformer_layer_spec` | `def(self, vp_stage)` | [`verl/models/mcore/model_initializer.py`](../../verl/models/mcore/model_initializer.py#L135) |
| `MixtralModel.initialize` | `def(self, **kwargs)` | [`verl/models/mcore/model_initializer.py`](../../verl/models/mcore/model_initializer.py#L141) |
| `mock_policy` | `def(traj_id, step, obs)` | [`tests/ray_cpu/test_async_rollout_equivalence.py`](../../tests/ray_cpu/test_async_rollout_equivalence.py#L55) |
| `ModelActor.__init__` | `def(self)` | [`tests/ray_gpu/test_driverfunc_to_worker.py`](../../tests/ray_gpu/test_driverfunc_to_worker.py#L32) |
| `ModelConfig.__init__` | `def(self)` | [`scripts/converter_hf_to_mcore.py`](../../scripts/converter_hf_to_mcore.py#L49) |
| `ModelMergerConfig.__post_init__` | `def(self)` | [`scripts/model_merger.py`](../../scripts/model_merger.py#L88) |
| `ModelRegistry.get_supported_archs` | `def()` | [`verl/models/registry.py`](../../verl/models/registry.py#L57) |
| `ModelRegistry.load_model_cls` | `def(model_arch, value)` | [`verl/models/registry.py`](../../verl/models/registry.py#L41) |
| `MultiTaskEnvironmentManager.__init__` | `def(self, managers, task_max_steps, config)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L622) |
| `MultiTaskEnvironmentManager._done_infos` | `def(self, task)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L757) |
| `MultiTaskEnvironmentManager._merge_arrays` | `def(self, task_values, batch_size, dtype)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L792) |
| `MultiTaskEnvironmentManager._merge_infos` | `def(self, task_infos, batch_size)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L785) |
| `MultiTaskEnvironmentManager._merge_observations` | `def(self, task_obs, batch_size)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L766) |
| `MultiTaskEnvironmentManager._slice_optional_batch` | `def(value, indices)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L800) |
| `MultiTaskEnvironmentManager.close` | `def(self)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L836) |
| `MultiTaskEnvironmentManager.reset` | `def(self, kwargs)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L632) |
| `MultiTaskEnvironmentManager.step` | `def(self, text_actions)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L691) |
| `MultiTaskEnvironmentManager.success_evaluator` | `def(self, *args, **kwargs)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L809) |
| `MultiTurnSFTDataset.__getitem__` | `def(self, item)` | [`verl/utils/dataset/multiturn_sft_dataset.py`](../../verl/utils/dataset/multiturn_sft_dataset.py#L81) |
| `MultiTurnSFTDataset.__init__` | `def(self, parquet_files, tokenizer, config)` | [`verl/utils/dataset/multiturn_sft_dataset.py`](../../verl/utils/dataset/multiturn_sft_dataset.py#L34) |
| `MultiTurnSFTDataset.__len__` | `def(self)` | [`verl/utils/dataset/multiturn_sft_dataset.py`](../../verl/utils/dataset/multiturn_sft_dataset.py#L78) |
| `MultiTurnSFTDataset._download` | `def(self)` | [`verl/utils/dataset/multiturn_sft_dataset.py`](../../verl/utils/dataset/multiturn_sft_dataset.py#L56) |
| `MultiTurnSFTDataset._read_files_and_process` | `def(self)` | [`verl/utils/dataset/multiturn_sft_dataset.py`](../../verl/utils/dataset/multiturn_sft_dataset.py#L60) |
| `MultiTurnSFTDataset._read_files_and_process.series_to_item` | `def(ls)` | [`verl/utils/dataset/multiturn_sft_dataset.py`](../../verl/utils/dataset/multiturn_sft_dataset.py#L61) |
| `NaiveRewardManager.__call__` | `def(self, data, return_dict)` | [`verl/workers/reward_manager/naive.py`](../../verl/workers/reward_manager/naive.py#L32) |
| `NaiveRewardManager.__init__` | `def(self, tokenizer, num_examine, compute_score, reward_fn_key)` | [`verl/workers/reward_manager/naive.py`](../../verl/workers/reward_manager/naive.py#L26) |
| `NaiveRollout.__init__` | `def(self, module, config)` | [`verl/workers/rollout/naive/naive_rollout.py`](../../verl/workers/rollout/naive/naive_rollout.py#L37) |
| `NaiveRollout.generate_sequences` | `def(self, prompts)` | [`verl/workers/rollout/naive/naive_rollout.py`](../../verl/workers/rollout/naive/naive_rollout.py#L51) |
| `NCCLIDStore.__init__` | `def(self, nccl_id)` | [`verl/utils/rendezvous/ray_backend.py`](../../verl/utils/rendezvous/ray_backend.py#L25) |
| `NCCLIDStore.get` | `def(self)` | [`verl/utils/rendezvous/ray_backend.py`](../../verl/utils/rendezvous/ray_backend.py#L28) |
| `NestedNamespace.__init__` | `def(self, dictionary, **kwargs)` | [`verl/utils/py_functional.py`](../../verl/utils/py_functional.py#L194) |
| `NoLoggerWarnings` | `def()` | [`agent_system/environments/env_package/sokoban/sokoban/utils.py`](../../agent_system/environments/env_package/sokoban/sokoban/utils.py#L47) |
| `normalize` | `def(answer, pi)` | [`verl/utils/reward_score/prime_math/grader.py`](../../verl/utils/reward_score/prime_math/grader.py#L122) |
| `normalize_answer` | `def(answer)` | [`verl/utils/reward_score/prime_math/math_normalize.py`](../../verl/utils/reward_score/prime_math/math_normalize.py#L44) |
| `normalize_answer` | `def(s)` | [`verl/utils/reward_score/search_r1_like_qa_em.py`](../../verl/utils/reward_score/search_r1_like_qa_em.py#L23) |
| `normalize_answer.lower` | `def(text)` | [`verl/utils/reward_score/search_r1_like_qa_em.py`](../../verl/utils/reward_score/search_r1_like_qa_em.py#L34) |
| `normalize_answer.remove_articles` | `def(text)` | [`verl/utils/reward_score/search_r1_like_qa_em.py`](../../verl/utils/reward_score/search_r1_like_qa_em.py#L24) |
| `normalize_answer.remove_punc` | `def(text)` | [`verl/utils/reward_score/search_r1_like_qa_em.py`](../../verl/utils/reward_score/search_r1_like_qa_em.py#L30) |
| `normalize_answer.white_space_fix` | `def(text)` | [`verl/utils/reward_score/search_r1_like_qa_em.py`](../../verl/utils/reward_score/search_r1_like_qa_em.py#L27) |
| `normalize_color` | `def(color_string)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/normalize.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/normalize.py#L57) |
| `normalize_color_size` | `def(product_prices)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/normalize.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/normalize.py#L64) |
| `normalize_final_answer` | `def(final_answer)` | [`verl/utils/reward_score/math_dapo.py`](../../verl/utils/reward_score/math_dapo.py#L125) |
| `normalize_model_name` | `def(name, pp_rank, vpp_rank, transformer_config, layer_name)` | [`verl/utils/model.py`](../../verl/utils/model.py#L208) |
| `normalize_pp_vpp_params` | `def(params, num_hidden_layers, layer_name)` | [`verl/utils/model.py`](../../verl/utils/model.py#L232) |
| `normalize_task_name` | `def(task_name)` | [`verl/trainer/ppo/metric_utils.py`](../../verl/trainer/ppo/metric_utils.py#L58) |
| `now` | `def()` | [`verl/utils/gpu_profiler.py`](../../verl/utils/gpu_profiler.py#L87) |
| `NumberLineEnv.__init__` | `def(self, max_position)` | [`agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/numberline.py`](../../agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/numberline.py#L45) |
| `NumberLineEnv._get_observation` | `def(self)` | [`agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/numberline.py`](../../agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/numberline.py#L109) |
| `NumberLineEnv.close` | `def(self)` | [`agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/numberline.py`](../../agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/numberline.py#L120) |
| `NumberLineEnv.reset` | `def(self, seed, options)` | [`agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/numberline.py`](../../agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/numberline.py#L92) |
| `NumberLineEnv.step` | `def(self, action)` | [`agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/numberline.py`](../../agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/numberline.py#L61) |
| `numpy_to_torch` | `def(array, device)` | [`agent_system/multi_turn_rollout/utils.py`](../../agent_system/multi_turn_rollout/utils.py#L52) |
| `NVMegatronRayWorkerGroup.__init__` | `def(self, resource_pool, ray_cls_with_init, **kwargs)` | [`verl/single_controller/ray/megatron.py`](../../verl/single_controller/ray/megatron.py#L32) |
| `offload_fsdp2_model_to_cpu` | `def(model, empty_cache)` | [`verl/utils/fsdp_utils.py`](../../verl/utils/fsdp_utils.py#L153) |
| `offload_fsdp_model_to_cpu` | `def(model, empty_cache)` | [`verl/utils/fsdp_utils.py`](../../verl/utils/fsdp_utils.py#L130) |
| `offload_fsdp_optimizer` | `def(optimizer)` | [`verl/utils/fsdp_utils.py`](../../verl/utils/fsdp_utils.py#L188) |
| `offload_megatron_copy_params` | `def(optimizers)` | [`verl/utils/megatron_utils.py`](../../verl/utils/megatron_utils.py#L303) |
| `offload_megatron_copy_params._iter_opts` | `def(opt)` | [`verl/utils/megatron_utils.py`](../../verl/utils/megatron_utils.py#L312) |
| `offload_megatron_copy_params.offload_group_to_cpu` | `def(group)` | [`verl/utils/megatron_utils.py`](../../verl/utils/megatron_utils.py#L322) |
| `offload_megatron_copy_params.offload_tensor_to_cpu` | `def(tensor)` | [`verl/utils/megatron_utils.py`](../../verl/utils/megatron_utils.py#L317) |
| `offload_megatron_model_to_cpu` | `def(models)` | [`verl/utils/megatron_utils.py`](../../verl/utils/megatron_utils.py#L240) |
| `offload_megatron_optimizer` | `def(optimizers)` | [`verl/utils/megatron_utils.py`](../../verl/utils/megatron_utils.py#L385) |
| `offload_megatron_optimizer._iter_opts` | `def(opt)` | [`verl/utils/megatron_utils.py`](../../verl/utils/megatron_utils.py#L386) |
| `OffloadHandler.__init__` | `def(self)` | [`verl/utils/activation_offload.py`](../../verl/utils/activation_offload.py#L92) |
| `OffloadHandler.tensor_pop` | `def(self, tensor_tag, **kwargs)` | [`verl/utils/activation_offload.py`](../../verl/utils/activation_offload.py#L99) |
| `OffloadHandler.tensor_push` | `def(self, tensor, **kwargs)` | [`verl/utils/activation_offload.py`](../../verl/utils/activation_offload.py#L95) |
| `only_int_check` | `def(val)` | [`verl/utils/reward_score/prime_code/testing_util.py`](../../verl/utils/reward_score/prime_code/testing_util.py#L68) |
| `OPDRayTrainer.__init__` | `def(self, *args, **kwargs)` | [`verl/trainer/ppo/opd_ray_trainer.py`](../../verl/trainer/ppo/opd_ray_trainer.py#L122) |
| `OPDRayTrainer._save_checkpoint` | `def(self)` | [`verl/trainer/ppo/opd_ray_trainer.py`](../../verl/trainer/ppo/opd_ray_trainer.py#L230) |
| `OPDRayTrainer.compute_teacher_log_probs` | `def(self, batch)` | [`verl/trainer/ppo/opd_ray_trainer.py`](../../verl/trainer/ppo/opd_ray_trainer.py#L252) |
| `OPDRayTrainer.fit` | `def(self)` | [`verl/trainer/ppo/opd_ray_trainer.py`](../../verl/trainer/ppo/opd_ray_trainer.py#L324) |
| `OPDRayTrainer.init_workers` | `def(self)` | [`verl/trainer/ppo/opd_ray_trainer.py`](../../verl/trainer/ppo/opd_ray_trainer.py#L146) |
| `OPDTaskRunner.run` | `def(self, config)` | [`verl/trainer/main_opd.py`](../../verl/trainer/main_opd.py#L39) |
| `OpenAIFunctionCallSchema.from_openai_function_parsed_schema` | `def(parsed_schema)` | [`verl/tools/schemas.py`](../../verl/tools/schemas.py#L67) |
| `pad_2d_list_to_length` | `def(response, pad_token_id, max_length)` | [`verl/utils/torch_functional.py`](../../verl/utils/torch_functional.py#L269) |
| `pad_dataproto_to_divisor` | `def(data, size_divisor)` | [`verl/protocol.py`](../../verl/protocol.py#L70) |
| `pad_packed_inputs` | `def(unpad_tokens, cu_seqlens, max_seqlen_in_batch, size)` | [`verl/utils/model.py`](../../verl/utils/model.py#L365) |
| `pad_sequence_to_length` | `def(tensors, max_seq_len, pad_token_id, left_pad)` | [`verl/utils/torch_functional.py`](../../verl/utils/torch_functional.py#L280) |
| `pad_to_sequence_parallel` | `def(unpad_tokens)` | [`verl/utils/megatron/sequence_parallel.py`](../../verl/utils/megatron/sequence_parallel.py#L29) |
| `parallel_compute_score_async` | `async def(evaluation_func, completions, references, tasks, extra_info, num_processes)` | [`verl/workers/reward_manager/prime.py`](../../verl/workers/reward_manager/prime.py#L44) |
| `parallel_init_module_fn` | `def(module, shard_states)` | [`verl/utils/fsdp_utils.py`](../../verl/utils/fsdp_utils.py#L299) |
| `parallel_init_module_fn.create_and_sync_state` | `def(param_name, state, is_param)` | [`verl/utils/fsdp_utils.py`](../../verl/utils/fsdp_utils.py#L320) |
| `parallel_init_module_fn.init_fn` | `def(sub_mod, recurse)` | [`verl/utils/fsdp_utils.py`](../../verl/utils/fsdp_utils.py#L339) |
| `parallel_load_safetensors` | `def(filepath)` | [`verl/utils/fsdp_utils.py`](../../verl/utils/fsdp_utils.py#L243) |
| `parallel_put` | `def(data_list, max_workers)` | [`verl/utils/ray_utils.py`](../../verl/utils/ray_utils.py#L49) |
| `parallel_put.put_data` | `def(index, data)` | [`verl/utils/ray_utils.py`](../../verl/utils/ray_utils.py#L64) |
| `ParallelLlamaAttention.__init__` | `def(self, config, megatron_config)` | [`verl/models/llama/megatron/layers/parallel_attention.py`](../../verl/models/llama/megatron/layers/parallel_attention.py#L172) |
| `ParallelLlamaAttention._init_rope` | `def(self)` | [`verl/models/llama/megatron/layers/parallel_attention.py`](../../verl/models/llama/megatron/layers/parallel_attention.py#L232) |
| `ParallelLlamaAttention._shape` | `def(self, tensor, seq_len, bsz)` | [`verl/models/llama/megatron/layers/parallel_attention.py`](../../verl/models/llama/megatron/layers/parallel_attention.py#L267) |
| `ParallelLlamaAttention.forward` | `def(self, hidden_states, attention_mask, position_ids)` | [`verl/models/llama/megatron/layers/parallel_attention.py`](../../verl/models/llama/megatron/layers/parallel_attention.py#L270) |
| `ParallelLlamaAttentionRmPad.forward` | `def(self, hidden_states, position_ids, sequence_length, indices, cu_seqlens, max_seqlen_in_batch)` | [`verl/models/llama/megatron/layers/parallel_attention.py`](../../verl/models/llama/megatron/layers/parallel_attention.py#L351) |
| `ParallelLlamaDecoderLayer.__init__` | `def(self, config, megatron_config, layer_idx)` | [`verl/models/llama/megatron/layers/parallel_decoder.py`](../../verl/models/llama/megatron/layers/parallel_decoder.py#L36) |
| `ParallelLlamaDecoderLayer.forward` | `def(self, hidden_states, attention_mask, position_ids)` | [`verl/models/llama/megatron/layers/parallel_decoder.py`](../../verl/models/llama/megatron/layers/parallel_decoder.py#L47) |
| `ParallelLlamaDecoderLayerRmPad.__init__` | `def(self, config, megatron_config, layer_idx)` | [`verl/models/llama/megatron/layers/parallel_decoder.py`](../../verl/models/llama/megatron/layers/parallel_decoder.py#L103) |
| `ParallelLlamaDecoderLayerRmPad.forward` | `def(self, hidden_states, position_ids, sequence_length, indices, cu_seqlens, max_seqlen_in_batch)` | [`verl/models/llama/megatron/layers/parallel_decoder.py`](../../verl/models/llama/megatron/layers/parallel_decoder.py#L114) |
| `ParallelLlamaForCausalLM.__init__` | `def(self, config, megatron_config)` | [`verl/models/llama/megatron/modeling_llama_megatron.py`](../../verl/models/llama/megatron/modeling_llama_megatron.py#L154) |
| `ParallelLlamaForCausalLM.forward` | `def(self, input_ids, attention_mask, position_ids)` | [`verl/models/llama/megatron/modeling_llama_megatron.py`](../../verl/models/llama/megatron/modeling_llama_megatron.py#L174) |
| `ParallelLlamaForCausalLMRmPad.__init__` | `def(self, config, megatron_config)` | [`verl/models/llama/megatron/modeling_llama_megatron.py`](../../verl/models/llama/megatron/modeling_llama_megatron.py#L282) |
| `ParallelLlamaForCausalLMRmPad._forward_head` | `def(self, hidden_states)` | [`verl/models/llama/megatron/modeling_llama_megatron.py`](../../verl/models/llama/megatron/modeling_llama_megatron.py#L304) |
| `ParallelLlamaForCausalLMRmPad._init_head` | `def(self, config)` | [`verl/models/llama/megatron/modeling_llama_megatron.py`](../../verl/models/llama/megatron/modeling_llama_megatron.py#L290) |
| `ParallelLlamaForCausalLMRmPad.forward` | `def(self, input_ids, attention_mask, position_ids)` | [`verl/models/llama/megatron/modeling_llama_megatron.py`](../../verl/models/llama/megatron/modeling_llama_megatron.py#L311) |
| `ParallelLlamaForCausalLMRmPadPP.__init__` | `def(self, config, megatron_config, pre_process, post_process, share_embeddings_and_output_weights)` | [`verl/models/llama/megatron/modeling_llama_megatron.py`](../../verl/models/llama/megatron/modeling_llama_megatron.py#L516) |
| `ParallelLlamaForCausalLMRmPadPP._forward_head` | `def(self, hidden_states)` | [`verl/models/llama/megatron/modeling_llama_megatron.py`](../../verl/models/llama/megatron/modeling_llama_megatron.py#L561) |
| `ParallelLlamaForCausalLMRmPadPP._init_head` | `def(self, config)` | [`verl/models/llama/megatron/modeling_llama_megatron.py`](../../verl/models/llama/megatron/modeling_llama_megatron.py#L547) |
| `ParallelLlamaForCausalLMRmPadPP.forward` | `def(self)` | [`verl/models/llama/megatron/modeling_llama_megatron.py`](../../verl/models/llama/megatron/modeling_llama_megatron.py#L569) |
| `ParallelLlamaForCausalLMRmPadPP.set_input_tensor` | `def(self, input_tensor)` | [`verl/models/llama/megatron/modeling_llama_megatron.py`](../../verl/models/llama/megatron/modeling_llama_megatron.py#L536) |
| `ParallelLlamaForValueRmPad._forward_head` | `def(self, hidden_states)` | [`verl/models/llama/megatron/modeling_llama_megatron.py`](../../verl/models/llama/megatron/modeling_llama_megatron.py#L379) |
| `ParallelLlamaForValueRmPad._init_head` | `def(self, config)` | [`verl/models/llama/megatron/modeling_llama_megatron.py`](../../verl/models/llama/megatron/modeling_llama_megatron.py#L370) |
| `ParallelLlamaForValueRmPad.forward` | `def(self, input_ids, attention_mask, position_ids)` | [`verl/models/llama/megatron/modeling_llama_megatron.py`](../../verl/models/llama/megatron/modeling_llama_megatron.py#L386) |
| `ParallelLlamaForValueRmPadPP._forward_head` | `def(self, hidden_states)` | [`verl/models/llama/megatron/modeling_llama_megatron.py`](../../verl/models/llama/megatron/modeling_llama_megatron.py#L643) |
| `ParallelLlamaForValueRmPadPP._init_head` | `def(self, config)` | [`verl/models/llama/megatron/modeling_llama_megatron.py`](../../verl/models/llama/megatron/modeling_llama_megatron.py#L634) |
| `ParallelLlamaForValueRmPadPP.forward` | `def(self)` | [`verl/models/llama/megatron/modeling_llama_megatron.py`](../../verl/models/llama/megatron/modeling_llama_megatron.py#L650) |
| `ParallelLlamaMLP.__init__` | `def(self, config, megatron_config)` | [`verl/models/llama/megatron/layers/parallel_mlp.py`](../../verl/models/llama/megatron/layers/parallel_mlp.py#L31) |
| `ParallelLlamaMLP.forward` | `def(self, x)` | [`verl/models/llama/megatron/layers/parallel_mlp.py`](../../verl/models/llama/megatron/layers/parallel_mlp.py#L71) |
| `ParallelLlamaModel.__init__` | `def(self, config, megatron_config)` | [`verl/models/llama/megatron/modeling_llama_megatron.py`](../../verl/models/llama/megatron/modeling_llama_megatron.py#L82) |
| `ParallelLlamaModel._prepare_decoder_attention_mask` | `def(self, attention_mask, input_shape, inputs_embeds)` | [`verl/models/llama/megatron/modeling_llama_megatron.py`](../../verl/models/llama/megatron/modeling_llama_megatron.py#L97) |
| `ParallelLlamaModel.forward` | `def(self, input_ids, attention_mask, position_ids)` | [`verl/models/llama/megatron/modeling_llama_megatron.py`](../../verl/models/llama/megatron/modeling_llama_megatron.py#L115) |
| `ParallelLlamaModelRmPad.__init__` | `def(self, config, megatron_config)` | [`verl/models/llama/megatron/modeling_llama_megatron.py`](../../verl/models/llama/megatron/modeling_llama_megatron.py#L223) |
| `ParallelLlamaModelRmPad.forward` | `def(self, input_ids, position_ids, sequence_length, indices, cu_seqlens, max_seqlen_in_batch)` | [`verl/models/llama/megatron/modeling_llama_megatron.py`](../../verl/models/llama/megatron/modeling_llama_megatron.py#L238) |
| `ParallelLlamaModelRmPadPP.__init__` | `def(self, config, megatron_config, pre_process, post_process)` | [`verl/models/llama/megatron/modeling_llama_megatron.py`](../../verl/models/llama/megatron/modeling_llama_megatron.py#L412) |
| `ParallelLlamaModelRmPadPP.forward` | `def(self, input_ids, position_ids, sequence_length, indices, cu_seqlens, max_seqlen_in_batch)` | [`verl/models/llama/megatron/modeling_llama_megatron.py`](../../verl/models/llama/megatron/modeling_llama_megatron.py#L464) |
| `ParallelLlamaModelRmPadPP.set_input_tensor` | `def(self, input_tensor)` | [`verl/models/llama/megatron/modeling_llama_megatron.py`](../../verl/models/llama/megatron/modeling_llama_megatron.py#L454) |
| `ParallelLlamaRMSNorm.__init__` | `def(self, config, megatron_config)` | [`verl/models/llama/megatron/layers/parallel_rmsnorm.py`](../../verl/models/llama/megatron/layers/parallel_rmsnorm.py#L27) |
| `ParallelLlamaRMSNorm.forward` | `def(self, hidden_states)` | [`verl/models/llama/megatron/layers/parallel_rmsnorm.py`](../../verl/models/llama/megatron/layers/parallel_rmsnorm.py#L41) |
| `ParallelQwen2Attention.__init__` | `def(self, config, megatron_config)` | [`verl/models/qwen2/megatron/layers/parallel_attention.py`](../../verl/models/qwen2/megatron/layers/parallel_attention.py#L149) |
| `ParallelQwen2Attention._init_rope` | `def(self)` | [`verl/models/qwen2/megatron/layers/parallel_attention.py`](../../verl/models/qwen2/megatron/layers/parallel_attention.py#L211) |
| `ParallelQwen2Attention._shape` | `def(self, tensor, seq_len, bsz)` | [`verl/models/qwen2/megatron/layers/parallel_attention.py`](../../verl/models/qwen2/megatron/layers/parallel_attention.py#L218) |
| `ParallelQwen2Attention.forward` | `def(self, hidden_states, attention_mask, position_ids)` | [`verl/models/qwen2/megatron/layers/parallel_attention.py`](../../verl/models/qwen2/megatron/layers/parallel_attention.py#L221) |
| `ParallelQwen2AttentionRmPad.forward` | `def(self, hidden_states, position_ids, sequence_length, indices, cu_seqlens, max_seqlen_in_batch)` | [`verl/models/qwen2/megatron/layers/parallel_attention.py`](../../verl/models/qwen2/megatron/layers/parallel_attention.py#L297) |
| `ParallelQwen2DecoderLayer.__init__` | `def(self, config, megatron_config, layer_idx)` | [`verl/models/qwen2/megatron/layers/parallel_decoder.py`](../../verl/models/qwen2/megatron/layers/parallel_decoder.py#L36) |
| `ParallelQwen2DecoderLayer.forward` | `def(self, hidden_states, attention_mask, position_ids)` | [`verl/models/qwen2/megatron/layers/parallel_decoder.py`](../../verl/models/qwen2/megatron/layers/parallel_decoder.py#L47) |
| `ParallelQwen2DecoderLayerRmPad.__init__` | `def(self, config, megatron_config, layer_idx)` | [`verl/models/qwen2/megatron/layers/parallel_decoder.py`](../../verl/models/qwen2/megatron/layers/parallel_decoder.py#L103) |
| `ParallelQwen2DecoderLayerRmPad.forward` | `def(self, hidden_states, position_ids, sequence_length, indices, cu_seqlens, max_seqlen_in_batch)` | [`verl/models/qwen2/megatron/layers/parallel_decoder.py`](../../verl/models/qwen2/megatron/layers/parallel_decoder.py#L114) |
| `ParallelQwen2ForCausalLM.__init__` | `def(self, config, megatron_config)` | [`verl/models/qwen2/megatron/modeling_qwen2_megatron.py`](../../verl/models/qwen2/megatron/modeling_qwen2_megatron.py#L154) |
| `ParallelQwen2ForCausalLM.forward` | `def(self, input_ids, attention_mask, position_ids)` | [`verl/models/qwen2/megatron/modeling_qwen2_megatron.py`](../../verl/models/qwen2/megatron/modeling_qwen2_megatron.py#L174) |
| `ParallelQwen2ForCausalLMRmPad.__init__` | `def(self, config, megatron_config)` | [`verl/models/qwen2/megatron/modeling_qwen2_megatron.py`](../../verl/models/qwen2/megatron/modeling_qwen2_megatron.py#L282) |
| `ParallelQwen2ForCausalLMRmPad._forward_head` | `def(self, hidden_states)` | [`verl/models/qwen2/megatron/modeling_qwen2_megatron.py`](../../verl/models/qwen2/megatron/modeling_qwen2_megatron.py#L304) |
| `ParallelQwen2ForCausalLMRmPad._init_head` | `def(self, config)` | [`verl/models/qwen2/megatron/modeling_qwen2_megatron.py`](../../verl/models/qwen2/megatron/modeling_qwen2_megatron.py#L290) |
| `ParallelQwen2ForCausalLMRmPad.forward` | `def(self, input_ids, attention_mask, position_ids)` | [`verl/models/qwen2/megatron/modeling_qwen2_megatron.py`](../../verl/models/qwen2/megatron/modeling_qwen2_megatron.py#L311) |
| `ParallelQwen2ForCausalLMRmPadPP.__init__` | `def(self, config, megatron_config, pre_process, post_process, share_embeddings_and_output_weights)` | [`verl/models/qwen2/megatron/modeling_qwen2_megatron.py`](../../verl/models/qwen2/megatron/modeling_qwen2_megatron.py#L515) |
| `ParallelQwen2ForCausalLMRmPadPP._forward_head` | `def(self, hidden_states)` | [`verl/models/qwen2/megatron/modeling_qwen2_megatron.py`](../../verl/models/qwen2/megatron/modeling_qwen2_megatron.py#L608) |
| `ParallelQwen2ForCausalLMRmPadPP._init_head` | `def(self, config)` | [`verl/models/qwen2/megatron/modeling_qwen2_megatron.py`](../../verl/models/qwen2/megatron/modeling_qwen2_megatron.py#L547) |
| `ParallelQwen2ForCausalLMRmPadPP.forward` | `def(self)` | [`verl/models/qwen2/megatron/modeling_qwen2_megatron.py`](../../verl/models/qwen2/megatron/modeling_qwen2_megatron.py#L619) |
| `ParallelQwen2ForCausalLMRmPadPP.set_input_tensor` | `def(self, input_tensor)` | [`verl/models/qwen2/megatron/modeling_qwen2_megatron.py`](../../verl/models/qwen2/megatron/modeling_qwen2_megatron.py#L536) |
| `ParallelQwen2ForCausalLMRmPadPP.setup_embeddings_and_output_layer` | `def(self)` | [`verl/models/qwen2/megatron/modeling_qwen2_megatron.py`](../../verl/models/qwen2/megatron/modeling_qwen2_megatron.py#L562) |
| `ParallelQwen2ForCausalLMRmPadPP.shared_embedding_or_output_weight` | `def(self)` | [`verl/models/qwen2/megatron/modeling_qwen2_megatron.py`](../../verl/models/qwen2/megatron/modeling_qwen2_megatron.py#L601) |
| `ParallelQwen2ForValueRmPad._forward_head` | `def(self, hidden_states)` | [`verl/models/qwen2/megatron/modeling_qwen2_megatron.py`](../../verl/models/qwen2/megatron/modeling_qwen2_megatron.py#L379) |
| `ParallelQwen2ForValueRmPad._init_head` | `def(self, config)` | [`verl/models/qwen2/megatron/modeling_qwen2_megatron.py`](../../verl/models/qwen2/megatron/modeling_qwen2_megatron.py#L370) |
| `ParallelQwen2ForValueRmPad.forward` | `def(self, input_ids, attention_mask, position_ids)` | [`verl/models/qwen2/megatron/modeling_qwen2_megatron.py`](../../verl/models/qwen2/megatron/modeling_qwen2_megatron.py#L386) |
| `ParallelQwen2ForValueRmPadPP._forward_head` | `def(self, hidden_states)` | [`verl/models/qwen2/megatron/modeling_qwen2_megatron.py`](../../verl/models/qwen2/megatron/modeling_qwen2_megatron.py#L692) |
| `ParallelQwen2ForValueRmPadPP._init_head` | `def(self, config)` | [`verl/models/qwen2/megatron/modeling_qwen2_megatron.py`](../../verl/models/qwen2/megatron/modeling_qwen2_megatron.py#L683) |
| `ParallelQwen2ForValueRmPadPP.forward` | `def(self)` | [`verl/models/qwen2/megatron/modeling_qwen2_megatron.py`](../../verl/models/qwen2/megatron/modeling_qwen2_megatron.py#L699) |
| `ParallelQwen2MLP.__init__` | `def(self, config, megatron_config)` | [`verl/models/qwen2/megatron/layers/parallel_mlp.py`](../../verl/models/qwen2/megatron/layers/parallel_mlp.py#L31) |
| `ParallelQwen2MLP.forward` | `def(self, x)` | [`verl/models/qwen2/megatron/layers/parallel_mlp.py`](../../verl/models/qwen2/megatron/layers/parallel_mlp.py#L71) |
| `ParallelQwen2Model.__init__` | `def(self, config, megatron_config)` | [`verl/models/qwen2/megatron/modeling_qwen2_megatron.py`](../../verl/models/qwen2/megatron/modeling_qwen2_megatron.py#L82) |
| `ParallelQwen2Model._prepare_decoder_attention_mask` | `def(self, attention_mask, input_shape, inputs_embeds)` | [`verl/models/qwen2/megatron/modeling_qwen2_megatron.py`](../../verl/models/qwen2/megatron/modeling_qwen2_megatron.py#L97) |
| `ParallelQwen2Model.forward` | `def(self, input_ids, attention_mask, position_ids)` | [`verl/models/qwen2/megatron/modeling_qwen2_megatron.py`](../../verl/models/qwen2/megatron/modeling_qwen2_megatron.py#L115) |
| `ParallelQwen2ModelRmPad.__init__` | `def(self, config, megatron_config)` | [`verl/models/qwen2/megatron/modeling_qwen2_megatron.py`](../../verl/models/qwen2/megatron/modeling_qwen2_megatron.py#L223) |
| `ParallelQwen2ModelRmPad.forward` | `def(self, input_ids, position_ids, sequence_length, indices, cu_seqlens, max_seqlen_in_batch)` | [`verl/models/qwen2/megatron/modeling_qwen2_megatron.py`](../../verl/models/qwen2/megatron/modeling_qwen2_megatron.py#L238) |
| `ParallelQwen2ModelRmPadPP.__init__` | `def(self, config, megatron_config, pre_process, post_process)` | [`verl/models/qwen2/megatron/modeling_qwen2_megatron.py`](../../verl/models/qwen2/megatron/modeling_qwen2_megatron.py#L412) |
| `ParallelQwen2ModelRmPadPP.forward` | `def(self, input_ids, position_ids, sequence_length, indices, cu_seqlens, max_seqlen_in_batch)` | [`verl/models/qwen2/megatron/modeling_qwen2_megatron.py`](../../verl/models/qwen2/megatron/modeling_qwen2_megatron.py#L463) |
| `ParallelQwen2ModelRmPadPP.set_input_tensor` | `def(self, input_tensor)` | [`verl/models/qwen2/megatron/modeling_qwen2_megatron.py`](../../verl/models/qwen2/megatron/modeling_qwen2_megatron.py#L453) |
| `ParallelQwen2RMSNorm.__init__` | `def(self, config, megatron_config)` | [`verl/models/qwen2/megatron/layers/parallel_rmsnorm.py`](../../verl/models/qwen2/megatron/layers/parallel_rmsnorm.py#L27) |
| `ParallelQwen2RMSNorm.forward` | `def(self, hidden_states)` | [`verl/models/qwen2/megatron/layers/parallel_rmsnorm.py`](../../verl/models/qwen2/megatron/layers/parallel_rmsnorm.py#L41) |
| `parse_action` | `def(action)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/engine.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/engine.py#L117) |
| `parse_args` | `def()` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/test.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/test.py#L86) |
| `parse_args` | `def()` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/train_choice_il.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/train_choice_il.py#L213) |
| `parse_args` | `def()` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/train_rl.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/train_rl.py#L171) |
| `parse_args` | `def()` | [`scripts/diagnose.py`](../../scripts/diagnose.py#L263) |
| `parse_gamefile` | `def(infos)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L28) |
| `parse_gamefile` | `def(infos)` | [`recipe/hgpo/env_manager.py`](../../recipe/hgpo/env_manager.py#L13) |
| `parse_item_page_amz` | `def(asin, verbose)` | [`agent_system/environments/env_package/webshop/webshop/transfer/predict_help.py`](../../agent_system/environments/env_package/webshop/webshop/transfer/predict_help.py#L296) |
| `parse_item_page_ebay` | `def(asin, verbose)` | [`agent_system/environments/env_package/webshop/webshop/transfer/predict_help.py`](../../agent_system/environments/env_package/webshop/webshop/transfer/predict_help.py#L63) |
| `parse_item_page_ws` | `def(asin, query, page_num, options, verbose)` | [`agent_system/environments/env_package/webshop/webshop/transfer/predict_help.py`](../../agent_system/environments/env_package/webshop/webshop/transfer/predict_help.py#L188) |
| `parse_results_amz` | `def(query, page_num, verbose)` | [`agent_system/environments/env_package/webshop/webshop/transfer/predict_help.py`](../../agent_system/environments/env_package/webshop/webshop/transfer/predict_help.py#L262) |
| `parse_results_ebay` | `def(query, page_num, verbose)` | [`agent_system/environments/env_package/webshop/webshop/transfer/predict_help.py`](../../agent_system/environments/env_package/webshop/webshop/transfer/predict_help.py#L26) |
| `parse_results_ws` | `def(query, page_num, verbose)` | [`agent_system/environments/env_package/webshop/webshop/transfer/predict_help.py`](../../agent_system/environments/env_package/webshop/webshop/transfer/predict_help.py#L146) |
| `patch_forward_with_backends` | `def(model, use_fused_kernels, fused_kernels_backend)` | [`verl/models/transformers/monkey_patch.py`](../../verl/models/transformers/monkey_patch.py#L196) |
| `patch_vllm_moe_model_weight_loader` | `def(model)` | [`verl/utils/vllm_utils.py`](../../verl/utils/vllm_utils.py#L80) |
| `patch_vlm_for_ulysses_input_slicing` | `def(model_class)` | [`verl/models/transformers/monkey_patch.py`](../../verl/models/transformers/monkey_patch.py#L120) |
| `patch_vlm_for_ulysses_input_slicing._create_ulysses_wrapped_decoder_forward` | `def(original_forward)` | [`verl/models/transformers/monkey_patch.py`](../../verl/models/transformers/monkey_patch.py#L126) |
| `patch_vlm_for_ulysses_input_slicing._create_ulysses_wrapped_decoder_forward.ulysses_wrapped_decoder_forward` | `def(self, *args, **kwargs)` | [`verl/models/transformers/monkey_patch.py`](../../verl/models/transformers/monkey_patch.py#L127) |
| `per_gpu_util_between` | `def(t0, t1)` | [`verl/utils/gpu_profiler.py`](../../verl/utils/gpu_profiler.py#L349) |
| `per_tensor_generator` | `def(actor_module, model_config, weight_converter, transformer_config, layer_name_mapping, convert_qkv_gate_up_by_simple_split)` | [`verl/utils/megatron_utils.py`](../../verl/utils/megatron_utils.py#L727) |
| `per_tensor_generator.tensor_generator` | `def()` | [`verl/utils/megatron_utils.py`](../../verl/utils/megatron_utils.py#L739) |
| `perform_single_search_batch` | `def(retrieval_service_url, query_list, topk, concurrent_semaphore, timeout)` | [`verl/tools/utils/search_r1_like_utils.py`](../../verl/tools/utils/search_r1_like_utils.py#L143) |
| `permanent_seed` | `def(seed)` | [`agent_system/environments/env_package/sokoban/sokoban/utils.py`](../../agent_system/environments/env_package/sokoban/sokoban/utils.py#L11) |
| `place_boxes_and_player` | `def(room, num_boxes, second_player)` | [`agent_system/environments/env_package/sokoban/sokoban/room_utils.py`](../../agent_system/environments/env_package/sokoban/sokoban/room_utils.py#L364) |
| `plot_animation` | `def(imgs)` | [`agent_system/environments/env_package/sokoban/sokoban/room_utils.py`](../../agent_system/environments/env_package/sokoban/sokoban/room_utils.py#L114) |
| `plot_animation.init` | `def()` | [`agent_system/environments/env_package/sokoban/sokoban/room_utils.py`](../../agent_system/environments/env_package/sokoban/sokoban/room_utils.py#L125) |
| `plot_animation.update` | `def(i)` | [`agent_system/environments/env_package/sokoban/sokoban/room_utils.py`](../../agent_system/environments/env_package/sokoban/sokoban/room_utils.py#L128) |
| `Point24Env.__init__` | `def(self, treat_face_cards_as_10, target_points)` | [`agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/points.py`](../../agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/points.py#L82) |
| `Point24Env._card_num_to_str` | `def(self, num)` | [`agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/points.py`](../../agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/points.py#L146) |
| `Point24Env._evaluate_formula` | `def(self)` | [`agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/points.py`](../../agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/points.py#L163) |
| `Point24Env._generate_cards` | `def(self)` | [`agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/points.py`](../../agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/points.py#L137) |
| `Point24Env._get_observation` | `def(self)` | [`agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/points.py`](../../agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/points.py#L182) |
| `Point24Env._is_valid_action` | `def(self, action)` | [`agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/points.py`](../../agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/points.py#L152) |
| `Point24Env._terminate_step` | `def(self, reward, info_key, is_truncated)` | [`agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/points.py`](../../agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/points.py#L179) |
| `Point24Env.reset` | `def(self, seed, options)` | [`agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/points.py`](../../agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/points.py#L95) |
| `Point24Env.set_action_space` | `def(self)` | [`agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/points.py`](../../agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/points.py#L90) |
| `Point24Env.step` | `def(self, action)` | [`agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/points.py`](../../agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/points.py#L114) |
| `pooling` | `def(pooler_output, last_hidden_state, attention_mask, pooling_method)` | [`examples/search/retriever/retrieval_server.py`](../../examples/search/retriever/retrieval_server.py#L46) |
| `pop_phase` | `def(name)` | [`verl/utils/gpu_profiler.py`](../../verl/utils/gpu_profiler.py#L336) |
| `post_process_logits` | `def(input_ids, logits, temperature, top_k, top_p)` | [`verl/utils/torch_functional.py`](../../verl/utils/torch_functional.py#L445) |
| `postprocess_data` | `def(input_ids, attention_mask, max_length, pad_token_id, left_pad, truncation)` | [`verl/utils/torch_functional.py`](../../verl/utils/torch_functional.py#L293) |
| `postprocess_packed_seqs` | `def(output, packed_seq_params, attention_mask, batch_size, seq_len, post_process)` | [`verl/models/mcore/util.py`](../../verl/models/mcore/util.py#L81) |
| `PrecisionType.is_bf16` | `def(precision)` | [`verl/utils/torch_dtypes.py`](../../verl/utils/torch_dtypes.py#L59) |
| `PrecisionType.is_fp16` | `def(precision)` | [`verl/utils/torch_dtypes.py`](../../verl/utils/torch_dtypes.py#L51) |
| `PrecisionType.is_fp32` | `def(precision)` | [`verl/utils/torch_dtypes.py`](../../verl/utils/torch_dtypes.py#L55) |
| `PrecisionType.supported_type` | `def(precision)` | [`verl/utils/torch_dtypes.py`](../../verl/utils/torch_dtypes.py#L43) |
| `PrecisionType.supported_types` | `def()` | [`verl/utils/torch_dtypes.py`](../../verl/utils/torch_dtypes.py#L47) |
| `PrecisionType.to_dtype` | `def(precision)` | [`verl/utils/torch_dtypes.py`](../../verl/utils/torch_dtypes.py#L63) |
| `PrecisionType.to_str` | `def(precision)` | [`verl/utils/torch_dtypes.py`](../../verl/utils/torch_dtypes.py#L74) |
| `predict` | `def(obs, info, model, softmax, rule, bart_model)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/test.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/test.py#L28) |
| `predict` | `def(obs, info)` | [`agent_system/environments/env_package/webshop/webshop/transfer/app.py`](../../agent_system/environments/env_package/webshop/webshop/transfer/app.py#L126) |
| `prepare_decoder_attention_mask` | `def(attention_mask, input_shape, inputs_embeds)` | [`verl/utils/torch_functional.py`](../../verl/utils/torch_functional.py#L530) |
| `prepare_fa2_from_position_ids` | `def(query, key, value, position_ids)` | [`verl/models/transformers/qwen2_vl.py`](../../verl/models/transformers/qwen2_vl.py#L164) |
| `prepare_fsdp_model` | `def(model, world_size)` | [`tests/workers/rollout/test_hf_rollout.py`](../../tests/workers/rollout/test_hf_rollout.py#L72) |
| `prepare_input_dataproto` | `def(tokenizer, config, validate)` | [`tests/workers/rollout/test_hf_rollout.py`](../../tests/workers/rollout/test_hf_rollout.py#L48) |
| `prepare_inputs` | `def(tokenizer, prompts, max_prompt_length)` | [`tests/workers/rollout/utils_sglang.py`](../../tests/workers/rollout/utils_sglang.py#L95) |
| `preprocess` | `def(text)` | [`examples/data_preprocess/hellaswag.py`](../../examples/data_preprocess/hellaswag.py#L28) |
| `preprocess_packed_seqs` | `def(input_ids, attention_mask, pre_process)` | [`verl/models/mcore/util.py`](../../verl/models/mcore/util.py#L21) |
| `PrimeRewardManager.__call__` | `def(self, data, return_dict)` | [`verl/workers/reward_manager/prime.py`](../../verl/workers/reward_manager/prime.py#L146) |
| `PrimeRewardManager.__init__` | `def(self, tokenizer, num_examine, compute_score, reward_fn_key)` | [`verl/workers/reward_manager/prime.py`](../../verl/workers/reward_manager/prime.py#L102) |
| `PrimeRewardManager.verify` | `def(self, data)` | [`verl/workers/reward_manager/prime.py`](../../verl/workers/reward_manager/prime.py#L114) |
| `PRIMERewardModelWorker.__init__` | `def(self, config)` | [`recipe/prime/prime_fsdp_workers.py`](../../recipe/prime/prime_fsdp_workers.py#L51) |
| `PRIMERewardModelWorker._build_reward_ref_model_optimizer` | `def(self, config)` | [`recipe/prime/prime_fsdp_workers.py`](../../recipe/prime/prime_fsdp_workers.py#L84) |
| `PRIMERewardModelWorker.compute_rm_score` | `def(self, data)` | [`recipe/prime/prime_fsdp_workers.py`](../../recipe/prime/prime_fsdp_workers.py#L266) |
| `PRIMERewardModelWorker.init_model` | `def(self)` | [`recipe/prime/prime_fsdp_workers.py`](../../recipe/prime/prime_fsdp_workers.py#L236) |
| `PRIMERewardModelWorker.load_checkpoint` | `def(self, local_path, del_local_after_load)` | [`recipe/prime/prime_fsdp_workers.py`](../../recipe/prime/prime_fsdp_workers.py#L354) |
| `PRIMERewardModelWorker.save_checkpoint` | `def(self, local_path, hdfs_path, global_step, max_ckpt_to_keep)` | [`recipe/prime/prime_fsdp_workers.py`](../../recipe/prime/prime_fsdp_workers.py#L341) |
| `PRIMERewardModelWorker.update_rm` | `def(self, data)` | [`recipe/prime/prime_fsdp_workers.py`](../../recipe/prime/prime_fsdp_workers.py#L301) |
| `print_model_size` | `def(model, name)` | [`verl/utils/model.py`](../../verl/utils/model.py#L154) |
| `print_rank_0` | `def(message)` | [`verl/utils/megatron_utils.py`](../../verl/utils/megatron_utils.py#L422) |
| `process` | `def(s)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/train_choice_il.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/train_choice_il.py#L89) |
| `process` | `def(iter)` | [`verl/utils/debug/trajectory_tracker.py`](../../verl/utils/debug/trajectory_tracker.py#L93) |
| `process_dataset` | `def(input, output, max_len)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/train_search_il.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/train_search_il.py#L92) |
| `process_goal` | `def(state)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/train_choice_il.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/train_choice_il.py#L95) |
| `process_goal` | `def(state)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/train_search_il.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/train_search_il.py#L26) |
| `process_goal` | `def(state)` | [`agent_system/environments/env_package/webshop/webshop/transfer/app.py`](../../agent_system/environments/env_package/webshop/webshop/transfer/app.py#L34) |
| `process_image` | `def(image, max_pixels, min_pixels)` | [`agent_system/multi_turn_rollout/utils.py`](../../agent_system/multi_turn_rollout/utils.py#L62) |
| `process_image` | `def(image)` | [`verl/utils/dataset/vision_utils.py`](../../verl/utils/dataset/vision_utils.py#L23) |
| `process_item` | `def(config, data_source, response_lst, reward_data)` | [`recipe/r1/main_eval.py`](../../recipe/r1/main_eval.py#L33) |
| `process_item` | `def(reward_fn, data_source, response_lst, reward_data)` | [`verl/trainer/main_eval.py`](../../verl/trainer/main_eval.py#L33) |
| `process_position_ids` | `def(position_ids)` | [`verl/models/transformers/qwen2_vl.py`](../../verl/models/transformers/qwen2_vl.py#L395) |
| `process_single_row` | `def(row, current_split_name, row_index)` | [`examples/data_preprocess/preprocess_search_r1_dataset.py`](../../examples/data_preprocess/preprocess_search_r1_dataset.py#L38) |
| `process_str` | `def(s)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/train_search_il.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/train_search_il.py#L21) |
| `process_str` | `def(s)` | [`agent_system/environments/env_package/webshop/webshop/transfer/app.py`](../../agent_system/environments/env_package/webshop/webshop/transfer/app.py#L28) |
| `process_validation_metrics` | `def(data_sources, sample_inputs, infos_dict, seed)` | [`verl/trainer/ppo/metric_utils.py`](../../verl/trainer/ppo/metric_utils.py#L500) |
| `process_video` | `def(video, nframes, fps, fps_min_frames, fps_max_frames)` | [`verl/utils/dataset/vision_utils.py`](../../verl/utils/dataset/vision_utils.py#L62) |
| `profile` | `def(n)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L314) |
| `profile.decorator_with_name` | `def(func)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L321) |
| `profile.decorator_with_name.func_wrapper` | `def(*args, **kwargs)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L322) |
| `ProfileKV.__enter__` | `def(self)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L307) |
| `ProfileKV.__exit__` | `def(self, type, value, traceback)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L310) |
| `ProfileKV.__init__` | `def(self, n)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L304) |
| `Profiler.__init__` | `def(self, config)` | [`verl/utils/debug/profile.py`](../../verl/utils/debug/profile.py#L22) |
| `Profiler._validate` | `def(self)` | [`verl/utils/debug/profile.py`](../../verl/utils/debug/profile.py#L49) |
| `Profiler.check` | `def(self)` | [`verl/utils/debug/profile.py`](../../verl/utils/debug/profile.py#L58) |
| `Profiler.save` | `def(self)` | [`verl/utils/debug/profile.py`](../../verl/utils/debug/profile.py#L75) |
| `Profiler.start` | `def(self)` | [`verl/utils/debug/profile.py`](../../verl/utils/debug/profile.py#L61) |
| `Profiler.step` | `def(self)` | [`verl/utils/debug/profile.py`](../../verl/utils/debug/profile.py#L66) |
| `Profiler.stop` | `def(self)` | [`verl/utils/debug/profile.py`](../../verl/utils/debug/profile.py#L70) |
| `Profiler.stop_and_save` | `def(self)` | [`verl/utils/debug/profile.py`](../../verl/utils/debug/profile.py#L85) |
| `Profiler.stop_trace` | `def(self)` | [`verl/utils/debug/profile.py`](../../verl/utils/debug/profile.py#L90) |
| `push_phase` | `def(name)` | [`verl/utils/gpu_profiler.py`](../../verl/utils/gpu_profiler.py#L328) |
| `QKVParallelLinear.__init__` | `def(self, input_size, num_heads, num_key_value_heads, head_dim, **kwargs)` | [`verl/models/llama/megatron/layers/parallel_linear.py`](../../verl/models/llama/megatron/layers/parallel_linear.py#L21) |
| `QKVParallelLinear.__init__` | `def(self, input_size, num_heads, num_key_value_heads, head_dim, **kwargs)` | [`verl/models/qwen2/megatron/layers/parallel_linear.py`](../../verl/models/qwen2/megatron/layers/parallel_linear.py#L21) |
| `quick_task` | `def(x)` | [`tests/utils/cpu_tests/test_timeout_decorator.py`](../../tests/utils/cpu_tests/test_timeout_decorator.py#L30) |
| `Qwen25VLModel.get_transformer_layer_spec` | `def(self, vp_stage)` | [`verl/models/mcore/model_initializer.py`](../../verl/models/mcore/model_initializer.py#L209) |
| `Qwen25VLModel.initialize` | `def(self, pre_process, post_process, share_embeddings_and_output_weights, value, **extra_kwargs)` | [`verl/models/mcore/model_initializer.py`](../../verl/models/mcore/model_initializer.py#L214) |
| `qwen2_attn_forward` | `def(self, hidden_states, position_embeddings, attention_mask, past_key_value, cache_position, **kwargs)` | [`verl/models/transformers/qwen2.py`](../../verl/models/transformers/qwen2.py#L149) |
| `qwen2_flash_attn_forward` | `def(self, hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, cache_position, position_embeddings)` | [`verl/models/transformers/qwen2.py`](../../verl/models/transformers/qwen2.py#L33) |
| `qwen2_vl_attn_forward` | `def(self, hidden_states, attention_mask, position_ids, position_embeddings, **kwargs)` | [`verl/models/transformers/qwen2_vl.py`](../../verl/models/transformers/qwen2_vl.py#L268) |
| `qwen2_vl_base_forward` | `def(self, input_ids, attention_mask, labels, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw, **kwargs)` | [`verl/models/transformers/qwen2_vl.py`](../../verl/models/transformers/qwen2_vl.py#L414) |
| `qwen2_vl_forward` | `def(self, input_ids, attention_mask, position_ids, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw, **kwargs)` | [`verl/models/transformers/qwen2_vl.py`](../../verl/models/transformers/qwen2_vl.py#L431) |
| `Qwen2DynamicNTKScalingRotaryEmbedding.__init__` | `def(self, dim, max_position_embeddings, base, device, scaling_factor)` | [`verl/models/qwen2/megatron/layers/parallel_attention.py`](../../verl/models/qwen2/megatron/layers/parallel_attention.py#L98) |
| `Qwen2DynamicNTKScalingRotaryEmbedding._set_cos_sin_cache` | `def(self, seq_len, device, dtype)` | [`verl/models/qwen2/megatron/layers/parallel_attention.py`](../../verl/models/qwen2/megatron/layers/parallel_attention.py#L102) |
| `Qwen2LinearScalingRotaryEmbedding.__init__` | `def(self, dim, max_position_embeddings, base, device, scaling_factor)` | [`verl/models/qwen2/megatron/layers/parallel_attention.py`](../../verl/models/qwen2/megatron/layers/parallel_attention.py#L79) |
| `Qwen2LinearScalingRotaryEmbedding._set_cos_sin_cache` | `def(self, seq_len, device, dtype)` | [`verl/models/qwen2/megatron/layers/parallel_attention.py`](../../verl/models/qwen2/megatron/layers/parallel_attention.py#L83) |
| `Qwen2MoEModel.get_transformer_layer_spec` | `def(self, vp_stage)` | [`verl/models/mcore/model_initializer.py`](../../verl/models/mcore/model_initializer.py#L111) |
| `Qwen2MoEModel.initialize` | `def(self, **kwargs)` | [`verl/models/mcore/model_initializer.py`](../../verl/models/mcore/model_initializer.py#L122) |
| `Qwen2RotaryEmbedding.__init__` | `def(self, dim, max_position_embeddings, base, device)` | [`verl/models/qwen2/megatron/layers/parallel_attention.py`](../../verl/models/qwen2/megatron/layers/parallel_attention.py#L43) |
| `Qwen2RotaryEmbedding._set_cos_sin_cache` | `def(self, seq_len, device, dtype)` | [`verl/models/qwen2/megatron/layers/parallel_attention.py`](../../verl/models/qwen2/megatron/layers/parallel_attention.py#L55) |
| `Qwen2RotaryEmbedding.forward` | `def(self, x, seq_len)` | [`verl/models/qwen2/megatron/layers/parallel_attention.py`](../../verl/models/qwen2/megatron/layers/parallel_attention.py#L65) |
| `qwen3_vl_base_forward` | `def(self, input_ids, attention_mask, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw, **kwargs)` | [`verl/models/transformers/qwen3_vl.py`](../../verl/models/transformers/qwen3_vl.py#L234) |
| `Qwen3MoEModel.get_transformer_layer_spec` | `def(self, vp_stage)` | [`verl/models/mcore/model_initializer.py`](../../verl/models/mcore/model_initializer.py#L153) |
| `Qwen3MoEModel.initialize` | `def(self, **kwargs)` | [`verl/models/mcore/model_initializer.py`](../../verl/models/mcore/model_initializer.py#L159) |
| `random_idx` | `def(cum_weights)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/utils.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/utils.py#L20) |
| `RandomPolicy.__init__` | `def(self)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/models/models.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/models/models.py#L43) |
| `RandomPolicy.forward` | `def(self, observation, available_actions)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/models/models.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/models/models.py#L46) |
| `ray_init_shutdown` | `def()` | [`tests/ray_cpu/test_decorator.py`](../../tests/ray_cpu/test_decorator.py#L31) |
| `ray_noset_visible_devices` | `def(env_vars)` | [`verl/utils/ray_utils.py`](../../verl/utils/ray_utils.py#L26) |
| `RayClassWithInitArgs.__call__` | `def(self, placement_group, placement_group_bundle_idx, use_gpu, num_gpus, sharing_with, device_name)` | [`verl/single_controller/ray/base.py`](../../verl/single_controller/ray/base.py#L192) |
| `RayClassWithInitArgs.__init__` | `def(self, cls, *args, **kwargs)` | [`verl/single_controller/ray/base.py`](../../verl/single_controller/ray/base.py#L170) |
| `RayClassWithInitArgs.set_additional_resource` | `def(self, additional_resource)` | [`verl/single_controller/ray/base.py`](../../verl/single_controller/ray/base.py#L176) |
| `RayClassWithInitArgs.update_options` | `def(self, options)` | [`verl/single_controller/ray/base.py`](../../verl/single_controller/ray/base.py#L184) |
| `RayDAPOTrainer.fit` | `def(self)` | [`recipe/dapo/dapo_ray_trainer.py`](../../recipe/dapo/dapo_ray_trainer.py#L44) |
| `RayMultiProcessTestCase.setUp` | `def(self)` | [`tests/workers/rollout/test_sglang_async_rollout_sf_tools.py`](../../tests/workers/rollout/test_sglang_async_rollout_sf_tools.py#L384) |
| `RayMultiProcessTestCase.tearDown` | `def(self)` | [`tests/workers/rollout/test_sglang_async_rollout_sf_tools.py`](../../tests/workers/rollout/test_sglang_async_rollout_sf_tools.py#L390) |
| `RayPPOTrainer.__init__` | `def(self, config, tokenizer, role_worker_mapping, resource_pool_manager, ray_worker_group_cls, processor, reward_fn, val_reward_fn, train_dataset, val_dataset, collate_fn, train_sampler, device_name, traj_collector, envs, val_envs)` | [`recipe/hgpo/hgpo_ray_trainer.py`](../../recipe/hgpo/hgpo_ray_trainer.py#L413) |
| `RayPPOTrainer.__init__` | `def(self, config, tokenizer, role_worker_mapping, resource_pool_manager, ray_worker_group_cls, processor, reward_fn, val_reward_fn, train_dataset, val_dataset, collate_fn, train_sampler, device_name, traj_collector, envs, val_envs)` | [`verl/trainer/ppo/ray_trainer.py`](../../verl/trainer/ppo/ray_trainer.py#L438) |
| `RayPPOTrainer._attach_task_ids` | `def(batch)` | [`verl/trainer/ppo/ray_trainer.py`](../../verl/trainer/ppo/ray_trainer.py#L739) |
| `RayPPOTrainer._balance_batch` | `def(self, batch, metrics, logging_prefix)` | [`recipe/hgpo/hgpo_ray_trainer.py`](../../recipe/hgpo/hgpo_ray_trainer.py#L1015) |
| `RayPPOTrainer._balance_batch` | `def(self, batch, metrics, logging_prefix)` | [`verl/trainer/ppo/ray_trainer.py`](../../verl/trainer/ppo/ray_trainer.py#L1200) |
| `RayPPOTrainer._create_dataloader` | `def(self, train_dataset, val_dataset, collate_fn, train_sampler)` | [`recipe/hgpo/hgpo_ray_trainer.py`](../../recipe/hgpo/hgpo_ray_trainer.py#L597) |
| `RayPPOTrainer._create_dataloader` | `def(self, train_dataset, val_dataset, collate_fn, train_sampler)` | [`verl/trainer/ppo/ray_trainer.py`](../../verl/trainer/ppo/ray_trainer.py#L621) |
| `RayPPOTrainer._dump_generations` | `def(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path)` | [`recipe/hgpo/hgpo_ray_trainer.py`](../../recipe/hgpo/hgpo_ray_trainer.py#L662) |
| `RayPPOTrainer._dump_generations` | `def(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path)` | [`verl/trainer/ppo/ray_trainer.py`](../../verl/trainer/ppo/ray_trainer.py#L686) |
| `RayPPOTrainer._entropy_loss_metrics` | `def(batch, entropys, response_masks, loss_agg_mode)` | [`verl/trainer/ppo/ray_trainer.py`](../../verl/trainer/ppo/ray_trainer.py#L759) |
| `RayPPOTrainer._fast_forward_env_schedules` | `def(self)` | [`verl/trainer/ppo/ray_trainer.py`](../../verl/trainer/ppo/ray_trainer.py#L1150) |
| `RayPPOTrainer._load_checkpoint` | `def(self)` | [`recipe/hgpo/hgpo_ray_trainer.py`](../../recipe/hgpo/hgpo_ray_trainer.py#L964) |
| `RayPPOTrainer._load_checkpoint` | `def(self)` | [`verl/trainer/ppo/ray_trainer.py`](../../verl/trainer/ppo/ray_trainer.py#L1099) |
| `RayPPOTrainer._maybe_log_val_generations` | `def(self, inputs, outputs, scores)` | [`recipe/hgpo/hgpo_ray_trainer.py`](../../recipe/hgpo/hgpo_ray_trainer.py#L686) |
| `RayPPOTrainer._maybe_log_val_generations` | `def(self, inputs, outputs, scores)` | [`verl/trainer/ppo/ray_trainer.py`](../../verl/trainer/ppo/ray_trainer.py#L710) |
| `RayPPOTrainer._normalize_task_name` | `def(task_name)` | [`verl/trainer/ppo/ray_trainer.py`](../../verl/trainer/ppo/ray_trainer.py#L735) |
| `RayPPOTrainer._save_checkpoint` | `def(self)` | [`recipe/hgpo/hgpo_ray_trainer.py`](../../recipe/hgpo/hgpo_ray_trainer.py#L932) |
| `RayPPOTrainer._save_checkpoint` | `def(self)` | [`verl/trainer/ppo/ray_trainer.py`](../../verl/trainer/ppo/ray_trainer.py#L1067) |
| `RayPPOTrainer._validate` | `def(self)` | [`recipe/hgpo/hgpo_ray_trainer.py`](../../recipe/hgpo/hgpo_ray_trainer.py#L710) |
| `RayPPOTrainer._validate` | `def(self)` | [`verl/trainer/ppo/ray_trainer.py`](../../verl/trainer/ppo/ray_trainer.py#L815) |
| `RayPPOTrainer._validate_config` | `def(self)` | [`recipe/hgpo/hgpo_ray_trainer.py`](../../recipe/hgpo/hgpo_ray_trainer.py#L484) |
| `RayPPOTrainer._validate_config` | `def(self)` | [`verl/trainer/ppo/ray_trainer.py`](../../verl/trainer/ppo/ray_trainer.py#L508) |
| `RayPPOTrainer._validate_config.check_mutually_exclusive` | `def(mbs, mbs_per_gpu, name)` | [`recipe/hgpo/hgpo_ray_trainer.py`](../../recipe/hgpo/hgpo_ray_trainer.py#L495) |
| `RayPPOTrainer._validate_config.check_mutually_exclusive` | `def(mbs, mbs_per_gpu, name)` | [`verl/trainer/ppo/ray_trainer.py`](../../verl/trainer/ppo/ray_trainer.py#L519) |
| `RayPPOTrainer._validation_kwargs_for_batch` | `def(self, batch)` | [`verl/trainer/ppo/ray_trainer.py`](../../verl/trainer/ppo/ray_trainer.py#L792) |
| `RayPPOTrainer._validation_task_name` | `def(self, batch)` | [`verl/trainer/ppo/ray_trainer.py`](../../verl/trainer/ppo/ray_trainer.py#L771) |
| `RayPPOTrainer.fit` | `def(self)` | [`recipe/hgpo/hgpo_ray_trainer.py`](../../recipe/hgpo/hgpo_ray_trainer.py#L1028) |
| `RayPPOTrainer.fit` | `def(self)` | [`verl/trainer/ppo/ray_trainer.py`](../../verl/trainer/ppo/ray_trainer.py#L1213) |
| `RayPPOTrainer.init_workers` | `def(self)` | [`recipe/hgpo/hgpo_ray_trainer.py`](../../recipe/hgpo/hgpo_ray_trainer.py#L849) |
| `RayPPOTrainer.init_workers` | `def(self)` | [`verl/trainer/ppo/ray_trainer.py`](../../verl/trainer/ppo/ray_trainer.py#L984) |
| `RayPRIMETrainer.__init__` | `def(self, config, tokenizer, role_worker_mapping, resource_pool_manager, ray_worker_group_cls, reward_fn, val_reward_fn)` | [`recipe/prime/prime_ray_trainer.py`](../../recipe/prime/prime_ray_trainer.py#L145) |
| `RayPRIMETrainer._create_dataloader` | `def(self, *args, **kwargs)` | [`recipe/prime/prime_ray_trainer.py`](../../recipe/prime/prime_ray_trainer.py#L173) |
| `RayPRIMETrainer._load_checkpoint` | `def(self)` | [`recipe/prime/prime_ray_trainer.py`](../../recipe/prime/prime_ray_trainer.py#L256) |
| `RayPRIMETrainer._save_checkpoint` | `def(self)` | [`recipe/prime/prime_ray_trainer.py`](../../recipe/prime/prime_ray_trainer.py#L223) |
| `RayPRIMETrainer._validate_config` | `def(self)` | [`recipe/prime/prime_ray_trainer.py`](../../recipe/prime/prime_ray_trainer.py#L169) |
| `RayPRIMETrainer.filter_and_downsample` | `def(self, scores, batch)` | [`recipe/prime/prime_ray_trainer.py`](../../recipe/prime/prime_ray_trainer.py#L485) |
| `RayPRIMETrainer.fit` | `def(self)` | [`recipe/prime/prime_ray_trainer.py`](../../recipe/prime/prime_ray_trainer.py#L305) |
| `RayResourcePool.__init__` | `def(self, process_on_nodes, use_gpu, name_prefix, max_colocate_count, detached, accelerator_type)` | [`verl/single_controller/ray/base.py`](../../verl/single_controller/ray/base.py#L86) |
| `RayResourcePool.get_placement_groups` | `def(self, strategy, name, device_name)` | [`verl/single_controller/ray/base.py`](../../verl/single_controller/ray/base.py#L103) |
| `RaySPINTrainer.__init__` | `def(self, config, tokenizer, role_worker_mapping, resource_pool_manager, ray_worker_group_cls, processor, reward_fn, val_reward_fn, train_dataset, val_dataset, collate_fn, train_sampler)` | [`recipe/spin/spin_trainer.py`](../../recipe/spin/spin_trainer.py#L356) |
| `RaySPINTrainer._balance_batch` | `def(self, batch, metrics, logging_prefix)` | [`recipe/spin/spin_trainer.py`](../../recipe/spin/spin_trainer.py#L902) |
| `RaySPINTrainer._create_dataloader` | `def(self, train_dataset, val_dataset, collate_fn, train_sampler)` | [`recipe/spin/spin_trainer.py`](../../recipe/spin/spin_trainer.py#L524) |
| `RaySPINTrainer._load_checkpoint` | `def(self)` | [`recipe/spin/spin_trainer.py`](../../recipe/spin/spin_trainer.py#L849) |
| `RaySPINTrainer._maybe_log_val_generations` | `def(self, inputs, outputs, scores)` | [`recipe/spin/spin_trainer.py`](../../recipe/spin/spin_trainer.py#L589) |
| `RaySPINTrainer._save_checkpoint` | `def(self)` | [`recipe/spin/spin_trainer.py`](../../recipe/spin/spin_trainer.py#L803) |
| `RaySPINTrainer._validate` | `def(self)` | [`recipe/spin/spin_trainer.py`](../../recipe/spin/spin_trainer.py#L613) |
| `RaySPINTrainer._validate_config` | `def(self)` | [`recipe/spin/spin_trainer.py`](../../recipe/spin/spin_trainer.py#L411) |
| `RaySPINTrainer._validate_config.check_mutually_exclusive` | `def(mbs, mbs_per_gpu, name)` | [`recipe/spin/spin_trainer.py`](../../recipe/spin/spin_trainer.py#L423) |
| `RaySPINTrainer.fit_dpo` | `def(self)` | [`recipe/spin/spin_trainer.py`](../../recipe/spin/spin_trainer.py#L920) |
| `RaySPINTrainer.init_workers` | `def(self)` | [`recipe/spin/spin_trainer.py`](../../recipe/spin/spin_trainer.py#L730) |
| `RaySPPOTrainer.__init__` | `def(self, config, tokenizer, role_worker_mapping, resource_pool_manager, ray_worker_group_cls, processor, reward_fn, val_reward_fn, train_dataset, val_dataset, collate_fn, train_sampler, device_name)` | [`recipe/sppo/sppo_ray_trainer.py`](../../recipe/sppo/sppo_ray_trainer.py#L75) |
| `RaySPPOTrainer.fit` | `def(self)` | [`recipe/sppo/sppo_ray_trainer.py`](../../recipe/sppo/sppo_ray_trainer.py#L121) |
| `RayWorkerGroup.__init__` | `def(self, resource_pool, ray_cls_with_init, bin_pack, name_prefix, detached, worker_names, worker_handles, ray_wait_register_center_timeout, device_name, **kwargs)` | [`verl/single_controller/ray/base.py`](../../verl/single_controller/ray/base.py#L238) |
| `RayWorkerGroup._execute_remote_single_worker` | `def(self, worker, method_name, *args, **kwargs)` | [`verl/single_controller/ray/base.py`](../../verl/single_controller/ray/base.py#L490) |
| `RayWorkerGroup._init_with_detached_workers` | `def(self, worker_names, worker_handles)` | [`verl/single_controller/ray/base.py`](../../verl/single_controller/ray/base.py#L300) |
| `RayWorkerGroup._init_with_resource_pool` | `def(self, resource_pool, ray_cls_with_init, bin_pack, detached)` | [`verl/single_controller/ray/base.py`](../../verl/single_controller/ray/base.py#L309) |
| `RayWorkerGroup._is_worker_alive` | `def(self, worker)` | [`verl/single_controller/ray/base.py`](../../verl/single_controller/ray/base.py#L288) |
| `RayWorkerGroup.execute_all` | `def(self, method_name, *args, **kwargs)` | [`verl/single_controller/ray/base.py`](../../verl/single_controller/ray/base.py#L548) |
| `RayWorkerGroup.execute_all_async` | `def(self, method_name, *args, **kwargs)` | [`verl/single_controller/ray/base.py`](../../verl/single_controller/ray/base.py#L574) |
| `RayWorkerGroup.execute_all_sync` | `def(self, method_name, *args, **kwargs)` | [`verl/single_controller/ray/base.py`](../../verl/single_controller/ray/base.py#L561) |
| `RayWorkerGroup.execute_rank_zero` | `def(self, method_name, *args, **kwargs)` | [`verl/single_controller/ray/base.py`](../../verl/single_controller/ray/base.py#L535) |
| `RayWorkerGroup.execute_rank_zero_async` | `def(self, method_name, *args, **kwargs)` | [`verl/single_controller/ray/base.py`](../../verl/single_controller/ray/base.py#L522) |
| `RayWorkerGroup.execute_rank_zero_sync` | `def(self, method_name, *args, **kwargs)` | [`verl/single_controller/ray/base.py`](../../verl/single_controller/ray/base.py#L509) |
| `RayWorkerGroup.from_detached` | `def(cls, name_prefix, worker_names, worker_handles, ray_cls_with_init)` | [`verl/single_controller/ray/base.py`](../../verl/single_controller/ray/base.py#L407) |
| `RayWorkerGroup.fuse` | `def(self, prefix_set)` | [`verl/single_controller/ray/base.py`](../../verl/single_controller/ray/base.py#L478) |
| `RayWorkerGroup.master_address` | `def(self)` | [`verl/single_controller/ray/base.py`](../../verl/single_controller/ray/base.py#L603) |
| `RayWorkerGroup.master_port` | `def(self)` | [`verl/single_controller/ray/base.py`](../../verl/single_controller/ray/base.py#L607) |
| `RayWorkerGroup.spawn` | `def(self, prefix_set)` | [`verl/single_controller/ray/base.py`](../../verl/single_controller/ray/base.py#L427) |
| `RayWorkerGroup.spawn._rebind_actor_methods` | `def(worker_group, actor_name)` | [`verl/single_controller/ray/base.py`](../../verl/single_controller/ray/base.py#L439) |
| `RayWorkerGroup.spawn_fused` | `def(self, prefix_set)` | [`verl/single_controller/ray/base.py`](../../verl/single_controller/ray/base.py#L461) |
| `RayWorkerGroup.worker_names` | `def(self)` | [`verl/single_controller/ray/base.py`](../../verl/single_controller/ray/base.py#L403) |
| `RayWorkerGroup.workers` | `def(self)` | [`verl/single_controller/ray/base.py`](../../verl/single_controller/ray/base.py#L611) |
| `RayWorkerGroup.world_size` | `def(self)` | [`verl/single_controller/ray/base.py`](../../verl/single_controller/ray/base.py#L615) |
| `RCDQN.__init__` | `def(self, vocab_size, embedding_dim, hidden_dim, arch, grad, embs, gru_embed, get_image, bert_path)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/models/rnn.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/models/rnn.py#L9) |
| `RCDQN.forward` | `def(self, state_batch, act_batch, value, q, act)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/models/rnn.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/models/rnn.py#L66) |
| `RCDQN.prepare` | `def(self, ids)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/models/rnn.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/models/rnn.py#L54) |
| `read_csv` | `def(fname)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L498) |
| `read_html_template` | `def(path)` | [`agent_system/environments/env_package/webshop/webshop/transfer/webshop_lite.py`](../../agent_system/environments/env_package/webshop/webshop/transfer/webshop_lite.py#L15) |
| `read_html_template` | `def(path)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/engine.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/engine.py#L111) |
| `read_json` | `def(fname)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L489) |
| `read_jsonl` | `def(file_path)` | [`examples/search/retriever/retrieval_server.py`](../../examples/search/retriever/retrieval_server.py#L22) |
| `read_tb` | `def(path)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L503) |
| `rearrange_micro_batches` | `def(batch, max_token_len, dp_group, num_batches_divided_by, same_micro_num_in_dp, min_num_micro_batch)` | [`verl/utils/seqlen_balancing.py`](../../verl/utils/seqlen_balancing.py#L229) |
| `recover_left_padding` | `def(result, attention_mask, original_attention_mask, origin_seqlen, post_process)` | [`verl/models/mcore/util.py`](../../verl/models/mcore/util.py#L171) |
| `reduce_metrics` | `def(metrics)` | [`verl/trainer/ppo/metric_utils.py`](../../verl/trainer/ppo/metric_utils.py#L38) |
| `reduce_metrics` | `def(metrics)` | [`verl/utils/metric/utils.py`](../../verl/utils/metric/utils.py#L23) |
| `register` | `def(dispatch_mode, execute_mode, blocking, materialize_futures)` | [`verl/single_controller/base/decorator.py`](../../verl/single_controller/base/decorator.py#L505) |
| `register.decorator` | `def(func)` | [`verl/single_controller/base/decorator.py`](../../verl/single_controller/base/decorator.py#L529) |
| `register.decorator.async_inner` | `async def(*args, **kwargs)` | [`verl/single_controller/base/decorator.py`](../../verl/single_controller/base/decorator.py#L537) |
| `register.decorator.inner` | `def(*args, **kwargs)` | [`verl/single_controller/base/decorator.py`](../../verl/single_controller/base/decorator.py#L531) |
| `register_dispatch_mode` | `def(dispatch_mode_name, dispatch_fn, collect_fn)` | [`verl/single_controller/base/decorator.py`](../../verl/single_controller/base/decorator.py#L447) |
| `reliability_guard` | `def(maximum_memory_bytes)` | [`verl/utils/reward_score/prime_code/testing_util.py`](../../verl/utils/reward_score/prime_code/testing_util.py#L556) |
| `remote_call_wg` | `def(worker_names)` | [`tests/ray_gpu/test_worker_group_basics.py`](../../tests/ray_gpu/test_worker_group_basics.py#L69) |
| `remove_boxed` | `def(s)` | [`verl/utils/reward_score/math.py`](../../verl/utils/reward_score/math.py#L49) |
| `remove_boxed` | `def(s)` | [`verl/utils/reward_score/math_dapo.py`](../../verl/utils/reward_score/math_dapo.py#L50) |
| `remove_boxed` | `def(s)` | [`verl/utils/reward_score/prime_math/__init__.py`](../../verl/utils/reward_score/prime_math/__init__.py#L297) |
| `remove_left_padding` | `def(input_ids, attention_mask, position_ids, sequence_parallel, pre_process)` | [`verl/models/mcore/util.py`](../../verl/models/mcore/util.py#L132) |
| `remove_pad_token` | `def(input_ids, attention_mask)` | [`verl/utils/torch_functional.py`](../../verl/utils/torch_functional.py#L363) |
| `remove_right_units` | `def(string)` | [`verl/utils/reward_score/math.py`](../../verl/utils/reward_score/math.py#L137) |
| `repeat_kv` | `def(hidden_states, n_rep)` | [`verl/models/llama/megatron/layers/parallel_attention.py`](../../verl/models/llama/megatron/layers/parallel_attention.py#L157) |
| `repeat_kv` | `def(hidden_states, n_rep)` | [`verl/models/qwen2/megatron/layers/parallel_attention.py`](../../verl/models/qwen2/megatron/layers/parallel_attention.py#L134) |
| `repeat_kv` | `def(hidden_states, n_rep)` | [`verl/models/transformers/kimi_vl.py`](../../verl/models/transformers/kimi_vl.py#L114) |
| `repeat_kv` | `def(hidden_states, n_rep)` | [`verl/models/transformers/monkey_patch.py`](../../verl/models/transformers/monkey_patch.py#L37) |
| `report_and_reset` | `def(label)` | [`verl/utils/gpu_profiler.py`](../../verl/utils/gpu_profiler.py#L355) |
| `reset` | `def()` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L432) |
| `reset_dispatch_registry` | `def()` | [`tests/single_controller/base/test_decorator.py`](../../tests/single_controller/base/test_decorator.py#L22) |
| `ResourcePool.__call__` | `def(self)` | [`verl/single_controller/base/worker_group.py`](../../verl/single_controller/base/worker_group.py#L56) |
| `ResourcePool.__init__` | `def(self, process_on_nodes, max_colocate_count, n_gpus_per_node)` | [`verl/single_controller/base/worker_group.py`](../../verl/single_controller/base/worker_group.py#L34) |
| `ResourcePool.add_node` | `def(self, process_count)` | [`verl/single_controller/base/worker_group.py`](../../verl/single_controller/base/worker_group.py#L48) |
| `ResourcePool.local_rank_list` | `def(self)` | [`verl/single_controller/base/worker_group.py`](../../verl/single_controller/base/worker_group.py#L68) |
| `ResourcePool.local_world_size_list` | `def(self)` | [`verl/single_controller/base/worker_group.py`](../../verl/single_controller/base/worker_group.py#L63) |
| `ResourcePool.store` | `def(self)` | [`verl/single_controller/base/worker_group.py`](../../verl/single_controller/base/worker_group.py#L60) |
| `ResourcePool.world_size` | `def(self)` | [`verl/single_controller/base/worker_group.py`](../../verl/single_controller/base/worker_group.py#L52) |
| `ResourcePoolManager._check_resource_available` | `def(self)` | [`recipe/hgpo/hgpo_ray_trainer.py`](../../recipe/hgpo/hgpo_ray_trainer.py#L131) |
| `ResourcePoolManager._check_resource_available` | `def(self)` | [`recipe/spin/spin_trainer.py`](../../recipe/spin/spin_trainer.py#L105) |
| `ResourcePoolManager._check_resource_available` | `def(self)` | [`verl/trainer/ppo/ray_trainer.py`](../../verl/trainer/ppo/ray_trainer.py#L132) |
| `ResourcePoolManager.create_resource_pool` | `def(self)` | [`recipe/hgpo/hgpo_ray_trainer.py`](../../recipe/hgpo/hgpo_ray_trainer.py#L112) |
| `ResourcePoolManager.create_resource_pool` | `def(self)` | [`recipe/spin/spin_trainer.py`](../../recipe/spin/spin_trainer.py#L84) |
| `ResourcePoolManager.create_resource_pool` | `def(self)` | [`verl/trainer/ppo/ray_trainer.py`](../../verl/trainer/ppo/ray_trainer.py#L113) |
| `ResourcePoolManager.get_n_gpus` | `def(self)` | [`recipe/hgpo/hgpo_ray_trainer.py`](../../recipe/hgpo/hgpo_ray_trainer.py#L127) |
| `ResourcePoolManager.get_n_gpus` | `def(self)` | [`recipe/spin/spin_trainer.py`](../../recipe/spin/spin_trainer.py#L101) |
| `ResourcePoolManager.get_n_gpus` | `def(self)` | [`verl/trainer/ppo/ray_trainer.py`](../../verl/trainer/ppo/ray_trainer.py#L128) |
| `ResourcePoolManager.get_resource_pool` | `def(self, role)` | [`recipe/hgpo/hgpo_ray_trainer.py`](../../recipe/hgpo/hgpo_ray_trainer.py#L123) |
| `ResourcePoolManager.get_resource_pool` | `def(self, role)` | [`recipe/spin/spin_trainer.py`](../../recipe/spin/spin_trainer.py#L97) |
| `ResourcePoolManager.get_resource_pool` | `def(self, role)` | [`verl/trainer/ppo/ray_trainer.py`](../../verl/trainer/ppo/ray_trainer.py#L124) |
| `retrieve_endpoint` | `def(request)` | [`examples/search/retriever/retrieval_server.py`](../../examples/search/retriever/retrieval_server.py#L314) |
| `reverse_move` | `def(room_state, room_structure, box_mapping, last_pull, action)` | [`agent_system/environments/env_package/sokoban/sokoban/room_utils.py`](../../agent_system/environments/env_package/sokoban/sokoban/room_utils.py#L507) |
| `reverse_playing` | `def(room_state, room_structure, search_depth)` | [`agent_system/environments/env_package/sokoban/sokoban/room_utils.py`](../../agent_system/environments/env_package/sokoban/sokoban/room_utils.py#L414) |
| `reward_func` | `def(data_source, solution_str, ground_truth, extra_info)` | [`recipe/r1/reward_score.py`](../../recipe/r1/reward_score.py#L16) |
| `RewardManager.__call__` | `def(self, data, return_dict)` | [`examples/split_placement/main_ppo_split.py`](../../examples/split_placement/main_ppo_split.py#L42) |
| `RewardManager.__init__` | `def(self, tokenizer, num_examine)` | [`examples/split_placement/main_ppo_split.py`](../../examples/split_placement/main_ppo_split.py#L38) |
| `RewardModelWorker.__init__` | `def(self, config)` | [`recipe/spin/fsdp_workers.py`](../../recipe/spin/fsdp_workers.py#L277) |
| `RewardModelWorker.__init__` | `def(self, config)` | [`verl/workers/fsdp_workers.py`](../../verl/workers/fsdp_workers.py#L1216) |
| `RewardModelWorker.__init__` | `def(self, config)` | [`verl/workers/megatron_workers.py`](../../verl/workers/megatron_workers.py#L747) |
| `RewardModelWorker._build_model` | `def(self, config)` | [`recipe/spin/fsdp_workers.py`](../../recipe/spin/fsdp_workers.py#L308) |
| `RewardModelWorker._build_model` | `def(self, config)` | [`verl/workers/fsdp_workers.py`](../../verl/workers/fsdp_workers.py#L1246) |
| `RewardModelWorker._build_rm_model` | `def(self, model_path, override_model_config, override_transformer_config)` | [`verl/workers/megatron_workers.py`](../../verl/workers/megatron_workers.py#L783) |
| `RewardModelWorker._build_rm_model.megatron_rm_model_provider` | `def(pre_process, post_process)` | [`verl/workers/megatron_workers.py`](../../verl/workers/megatron_workers.py#L790) |
| `RewardModelWorker._expand_to_token_level` | `def(self, data, scores)` | [`recipe/spin/fsdp_workers.py`](../../recipe/spin/fsdp_workers.py#L430) |
| `RewardModelWorker._expand_to_token_level` | `def(self, data, scores)` | [`verl/workers/fsdp_workers.py`](../../verl/workers/fsdp_workers.py#L1375) |
| `RewardModelWorker._forward_micro_batch` | `def(self, micro_batch)` | [`recipe/spin/fsdp_workers.py`](../../recipe/spin/fsdp_workers.py#L374) |
| `RewardModelWorker._forward_micro_batch` | `def(self, micro_batch)` | [`verl/workers/fsdp_workers.py`](../../verl/workers/fsdp_workers.py#L1329) |
| `RewardModelWorker._switch_chat_template` | `def(self, data)` | [`recipe/spin/fsdp_workers.py`](../../recipe/spin/fsdp_workers.py#L445) |
| `RewardModelWorker._switch_chat_template` | `def(self, data)` | [`verl/workers/fsdp_workers.py`](../../verl/workers/fsdp_workers.py#L1390) |
| `RewardModelWorker.compute_rm_score` | `def(self, data)` | [`recipe/spin/fsdp_workers.py`](../../recipe/spin/fsdp_workers.py#L508) |
| `RewardModelWorker.compute_rm_score` | `def(self, data)` | [`verl/workers/fsdp_workers.py`](../../verl/workers/fsdp_workers.py#L1452) |
| `RewardModelWorker.compute_rm_score` | `def(self, data)` | [`verl/workers/megatron_workers.py`](../../verl/workers/megatron_workers.py#L872) |
| `RewardModelWorker.init_model` | `def(self)` | [`recipe/spin/fsdp_workers.py`](../../recipe/spin/fsdp_workers.py#L369) |
| `RewardModelWorker.init_model` | `def(self)` | [`verl/workers/fsdp_workers.py`](../../verl/workers/fsdp_workers.py#L1324) |
| `RewardModelWorker.init_model` | `def(self)` | [`verl/workers/megatron_workers.py`](../../verl/workers/megatron_workers.py#L826) |
| `RLHFDataset.__getitem__` | `def(self, item)` | [`verl/utils/dataset/rl_dataset.py`](../../verl/utils/dataset/rl_dataset.py#L248) |
| `RLHFDataset.__getstate__` | `def(self)` | [`verl/utils/dataset/rl_dataset.py`](../../verl/utils/dataset/rl_dataset.py#L363) |
| `RLHFDataset.__init__` | `def(self, data_files, tokenizer, config, processor)` | [`verl/utils/dataset/rl_dataset.py`](../../verl/utils/dataset/rl_dataset.py#L105) |
| `RLHFDataset.__len__` | `def(self)` | [`verl/utils/dataset/rl_dataset.py`](../../verl/utils/dataset/rl_dataset.py#L226) |
| `RLHFDataset._build_messages` | `def(self, example)` | [`verl/utils/dataset/rl_dataset.py`](../../verl/utils/dataset/rl_dataset.py#L229) |
| `RLHFDataset._download` | `def(self, use_origin_parquet)` | [`verl/utils/dataset/rl_dataset.py`](../../verl/utils/dataset/rl_dataset.py#L188) |
| `RLHFDataset._max_prompt_length_for_example` | `def(self, example)` | [`verl/utils/dataset/rl_dataset.py`](../../verl/utils/dataset/rl_dataset.py#L182) |
| `RLHFDataset._normalize_task_name` | `def(task_name)` | [`verl/utils/dataset/rl_dataset.py`](../../verl/utils/dataset/rl_dataset.py#L143) |
| `RLHFDataset._normalize_task_overrides` | `def(task_overrides)` | [`verl/utils/dataset/rl_dataset.py`](../../verl/utils/dataset/rl_dataset.py#L156) |
| `RLHFDataset._read_files_and_tokenize` | `def(self)` | [`verl/utils/dataset/rl_dataset.py`](../../verl/utils/dataset/rl_dataset.py#L195) |
| `RLHFDataset._task_config_value` | `def(self, example, key, default)` | [`verl/utils/dataset/rl_dataset.py`](../../verl/utils/dataset/rl_dataset.py#L173) |
| `RLHFDataset._task_name_from_example` | `def(self, example)` | [`verl/utils/dataset/rl_dataset.py`](../../verl/utils/dataset/rl_dataset.py#L165) |
| `RLHFDataset._truncation_for_example` | `def(self, example)` | [`verl/utils/dataset/rl_dataset.py`](../../verl/utils/dataset/rl_dataset.py#L185) |
| `RLHFDataset.resume_dataset_state` | `def(self)` | [`verl/utils/dataset/rl_dataset.py`](../../verl/utils/dataset/rl_dataset.py#L217) |
| `RLSDRayTrainer.__init__` | `def(self, *args, **kwargs)` | [`verl/trainer/ppo/rlsd_ray_trainer.py`](../../verl/trainer/ppo/rlsd_ray_trainer.py#L173) |
| `RLSDRayTrainer._compute_teacher_log_probs` | `def(self, batch)` | [`verl/trainer/ppo/rlsd_ray_trainer.py`](../../verl/trainer/ppo/rlsd_ray_trainer.py#L499) |
| `RLSDRayTrainer._get_rlsd_lambda` | `def(self, step)` | [`verl/trainer/ppo/rlsd_ray_trainer.py`](../../verl/trainer/ppo/rlsd_ray_trainer.py#L182) |
| `RLSDRayTrainer._save_checkpoint` | `def(self)` | [`verl/trainer/ppo/rlsd_ray_trainer.py`](../../verl/trainer/ppo/rlsd_ray_trainer.py#L188) |
| `RLSDRayTrainer.fit` | `def(self)` | [`verl/trainer/ppo/rlsd_ray_trainer.py`](../../verl/trainer/ppo/rlsd_ray_trainer.py#L207) |
| `RLSDTaskRunner.run` | `def(self, config)` | [`verl/trainer/main_rlsd.py`](../../verl/trainer/main_rlsd.py#L35) |
| `RMDataset.__getitem__` | `def(self, item)` | [`verl/utils/dataset/rm_dataset.py`](../../verl/utils/dataset/rm_dataset.py#L111) |
| `RMDataset.__init__` | `def(self, parquet_files, tokenizer, prompt_key, chosen_key, rejected_key, max_length, add_eos, cache_dir)` | [`verl/utils/dataset/rm_dataset.py`](../../verl/utils/dataset/rm_dataset.py#L40) |
| `RMDataset.__len__` | `def(self)` | [`verl/utils/dataset/rm_dataset.py`](../../verl/utils/dataset/rm_dataset.py#L96) |
| `RMDataset._download` | `def(self)` | [`verl/utils/dataset/rm_dataset.py`](../../verl/utils/dataset/rm_dataset.py#L70) |
| `RMDataset._download._download_files` | `def()` | [`verl/utils/dataset/rm_dataset.py`](../../verl/utils/dataset/rm_dataset.py#L71) |
| `RMDataset._pad_to_length` | `def(self, input_ids, attention_mask)` | [`verl/utils/dataset/rm_dataset.py`](../../verl/utils/dataset/rm_dataset.py#L99) |
| `RMDataset._read_files_and_tokenize` | `def(self)` | [`verl/utils/dataset/rm_dataset.py`](../../verl/utils/dataset/rm_dataset.py#L85) |
| `room_topology_generation` | `def(dim, p_change_directions, num_steps)` | [`agent_system/environments/env_package/sokoban/sokoban/room_utils.py`](../../agent_system/environments/env_package/sokoban/sokoban/room_utils.py#L287) |
| `rotate_half` | `def(x)` | [`verl/models/llama/megatron/layers/parallel_attention.py`](../../verl/models/llama/megatron/layers/parallel_attention.py#L142) |
| `rotate_half` | `def(x)` | [`verl/models/qwen2/megatron/layers/parallel_attention.py`](../../verl/models/qwen2/megatron/layers/parallel_attention.py#L119) |
| `rotate_half` | `def(x)` | [`verl/models/transformers/kimi_vl.py`](../../verl/models/transformers/kimi_vl.py#L70) |
| `roundup_divisible` | `def(a, b)` | [`verl/utils/seqlen_balancing.py`](../../verl/utils/seqlen_balancing.py#L225) |
| `run_episode` | `def(goal, env, verbose)` | [`agent_system/environments/env_package/webshop/webshop/transfer/app.py`](../../agent_system/environments/env_package/webshop/webshop/transfer/app.py#L136) |
| `run_generation` | `def(config)` | [`verl/trainer/main_generation.py`](../../verl/trainer/main_generation.py#L49) |
| `run_opd` | `def(config)` | [`verl/trainer/main_opd.py`](../../verl/trainer/main_opd.py#L20) |
| `run_ppo` | `def(config)` | [`recipe/dapo/main_dapo.py`](../../recipe/dapo/main_dapo.py#L34) |
| `run_ppo` | `def(config)` | [`recipe/hgpo/main_hgpo.py`](../../recipe/hgpo/main_hgpo.py#L33) |
| `run_ppo` | `def(config)` | [`recipe/spin/main_spin.py`](../../recipe/spin/main_spin.py#L30) |
| `run_ppo` | `def(config)` | [`recipe/sppo/main_sppo.py`](../../recipe/sppo/main_sppo.py#L36) |
| `run_ppo` | `def(config)` | [`verl/trainer/main_ppo.py`](../../verl/trainer/main_ppo.py#L34) |
| `run_prime` | `def(config, compute_score)` | [`recipe/prime/main_prime.py`](../../recipe/prime/main_prime.py#L43) |
| `run_reward_scoring` | `def(evaluation_func, completions, references, tasks, extra_info, num_processes)` | [`verl/workers/reward_manager/prime.py`](../../verl/workers/reward_manager/prime.py#L86) |
| `run_rlsd` | `def(config)` | [`verl/trainer/main_rlsd.py`](../../verl/trainer/main_rlsd.py#L16) |
| `run_sdar` | `def(config)` | [`verl/trainer/main_sdar.py`](../../verl/trainer/main_sdar.py#L17) |
| `run_skill_grpo` | `def(config)` | [`verl/trainer/main_skill_grpo.py`](../../verl/trainer/main_skill_grpo.py#L19) |
| `run_skillsd` | `def(config)` | [`verl/trainer/main_skillsd.py`](../../verl/trainer/main_skillsd.py#L16) |
| `run_target_and_put_in_queue` | `def(target_func, q)` | [`tests/utils/cpu_tests/test_timeout_decorator.py`](../../tests/utils/cpu_tests/test_timeout_decorator.py#L69) |
| `run_test` | `def(in_outs, test, debug, timeout)` | [`verl/utils/reward_score/prime_code/testing_util.py`](../../verl/utils/reward_score/prime_code/testing_util.py#L87) |
| `run_torch_entropy` | `def(hidden, weight, labels, reduction)` | [`tests/kernels/test_linear_cross_entropy.py`](../../tests/kernels/test_linear_cross_entropy.py#L45) |
| `run_verl_original_entropy` | `def(hidden, weight, labels)` | [`tests/kernels/test_linear_cross_entropy.py`](../../tests/kernels/test_linear_cross_entropy.py#L58) |
| `run_verl_torch_fused_entropy` | `def(hidden, weight, labels)` | [`tests/kernels/test_linear_cross_entropy.py`](../../tests/kernels/test_linear_cross_entropy.py#L70) |
| `Sandbox.__init__` | `def(self)` | [`tests/workers/rollout/test_vllm_tool_calling.py`](../../tests/workers/rollout/test_vllm_tool_calling.py#L53) |
| `Sandbox._start_fastapi_server` | `async def(self)` | [`tests/workers/rollout/test_vllm_tool_calling.py`](../../tests/workers/rollout/test_vllm_tool_calling.py#L80) |
| `Sandbox._start_fastapi_server.lifespan` | `async def(app)` | [`tests/workers/rollout/test_vllm_tool_calling.py`](../../tests/workers/rollout/test_vllm_tool_calling.py#L82) |
| `Sandbox.code_execution` | `async def(self, request)` | [`tests/workers/rollout/test_vllm_tool_calling.py`](../../tests/workers/rollout/test_vllm_tool_calling.py#L59) |
| `Sandbox.get_server_address` | `async def(self)` | [`tests/workers/rollout/test_vllm_tool_calling.py`](../../tests/workers/rollout/test_vllm_tool_calling.py#L98) |
| `SandboxFusionTool.__init__` | `def(self, config, tool_schema)` | [`verl/tools/sandbox_fusion_tools.py`](../../verl/tools/sandbox_fusion_tools.py#L106) |
| `SandboxFusionTool.calc_reward` | `async def(self, instance_id, **kwargs)` | [`verl/tools/sandbox_fusion_tools.py`](../../verl/tools/sandbox_fusion_tools.py#L175) |
| `SandboxFusionTool.create` | `async def(self, instance_id, ground_truth, **kwargs)` | [`verl/tools/sandbox_fusion_tools.py`](../../verl/tools/sandbox_fusion_tools.py#L144) |
| `SandboxFusionTool.execute` | `async def(self, instance_id, parameters, **kwargs)` | [`verl/tools/sandbox_fusion_tools.py`](../../verl/tools/sandbox_fusion_tools.py#L154) |
| `SandboxFusionTool.execute_code` | `def(self, instance_id, code, timeout, language)` | [`verl/tools/sandbox_fusion_tools.py`](../../verl/tools/sandbox_fusion_tools.py#L165) |
| `SandboxFusionTool.get_openai_tool_schema` | `def(self)` | [`verl/tools/sandbox_fusion_tools.py`](../../verl/tools/sandbox_fusion_tools.py#L141) |
| `SandboxFusionTool.release` | `async def(self, instance_id, **kwargs)` | [`verl/tools/sandbox_fusion_tools.py`](../../verl/tools/sandbox_fusion_tools.py#L178) |
| `save_to_hdfs` | `def(data, name, hdfs_dir, verbose)` | [`verl/utils/debug/trajectory_tracker.py`](../../verl/utils/debug/trajectory_tracker.py#L34) |
| `scoped_configure.__enter__` | `def(self)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L445) |
| `scoped_configure.__exit__` | `def(self, *args)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L449) |
| `scoped_configure.__init__` | `def(self, dir, format_strs)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L440) |
| `score` | `def(hand)` | [`agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/blackjack.py`](../../agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/blackjack.py#L71) |
| `SDARTaskRunner.run` | `def(self, config)` | [`verl/trainer/main_sdar.py`](../../verl/trainer/main_sdar.py#L36) |
| `search_projection` | `def(actions)` | [`agent_system/environments/env_package/search/projection.py`](../../agent_system/environments/env_package/search/projection.py#L34) |
| `search_results` | `def(data)` | [`agent_system/environments/env_package/webshop/webshop/transfer/webshop_lite.py`](../../agent_system/environments/env_package/webshop/webshop/transfer/webshop_lite.py#L25) |
| `search_results` | `def(session_id, keywords, page)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/app.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/app.py#L111) |
| `SearchEnvironmentManager.__init__` | `def(self, envs, projection_f, config)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L50) |
| `SearchEnvironmentManager.__init__` | `def(self, envs, projection_f, config)` | [`recipe/hgpo/env_manager.py`](../../recipe/hgpo/env_manager.py#L35) |
| `SearchEnvironmentManager._process_batch` | `def(self, batch_idx, total_batch_list, total_infos, success)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L120) |
| `SearchEnvironmentManager._process_batch` | `def(self, batch_idx, total_batch_list, total_infos, success)` | [`recipe/hgpo/env_manager.py`](../../recipe/hgpo/env_manager.py#L105) |
| `SearchEnvironmentManager.build_text_obs` | `def(self, text_obs, init)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L90) |
| `SearchEnvironmentManager.build_text_obs` | `def(self, text_obs, init)` | [`recipe/hgpo/env_manager.py`](../../recipe/hgpo/env_manager.py#L75) |
| `SearchEnvironmentManager.reset` | `def(self, kwargs)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L54) |
| `SearchEnvironmentManager.reset` | `def(self, kwargs)` | [`recipe/hgpo/env_manager.py`](../../recipe/hgpo/env_manager.py#L39) |
| `SearchEnvironmentManager.step` | `def(self, text_actions)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L68) |
| `SearchEnvironmentManager.step` | `def(self, text_actions)` | [`recipe/hgpo/env_manager.py`](../../recipe/hgpo/env_manager.py#L53) |
| `SearchExecutionWorker.__init__` | `def(self, enable_global_rate_limit, rate_limit)` | [`verl/tools/search_tool.py`](../../verl/tools/search_tool.py#L76) |
| `SearchExecutionWorker._init_rate_limit` | `def(self, rate_limit)` | [`verl/tools/search_tool.py`](../../verl/tools/search_tool.py#L79) |
| `SearchExecutionWorker.execute` | `def(self, fn, *fn_args, **fn_kwargs)` | [`verl/tools/search_tool.py`](../../verl/tools/search_tool.py#L87) |
| `SearchExecutionWorker.ping` | `def(self)` | [`verl/tools/search_tool.py`](../../verl/tools/search_tool.py#L83) |
| `SearchMemory.__getitem__` | `def(self, idx)` | [`agent_system/memory/memory.py`](../../agent_system/memory/memory.py#L115) |
| `SearchMemory.__init__` | `def(self)` | [`agent_system/memory/memory.py`](../../agent_system/memory/memory.py#L107) |
| `SearchMemory.__len__` | `def(self)` | [`agent_system/memory/memory.py`](../../agent_system/memory/memory.py#L112) |
| `SearchMemory.fetch` | `def(self, history_length, obs_key, action_key)` | [`agent_system/memory/memory.py`](../../agent_system/memory/memory.py#L142) |
| `SearchMemory.reset` | `def(self, batch_size)` | [`agent_system/memory/memory.py`](../../agent_system/memory/memory.py#L118) |
| `SearchMemory.store` | `def(self, record)` | [`agent_system/memory/memory.py`](../../agent_system/memory/memory.py#L125) |
| `SearchMultiProcessEnv.__del__` | `def(self)` | [`agent_system/environments/env_package/search/envs.py`](../../agent_system/environments/env_package/search/envs.py#L158) |
| `SearchMultiProcessEnv.__init__` | `def(self, seed, env_num, group_n, is_train, env_config)` | [`agent_system/environments/env_package/search/envs.py`](../../agent_system/environments/env_package/search/envs.py#L33) |
| `SearchMultiProcessEnv._sync_reset` | `def(self, env, kwargs)` | [`agent_system/environments/env_package/search/envs.py`](../../agent_system/environments/env_package/search/envs.py#L75) |
| `SearchMultiProcessEnv._sync_step` | `def(self, env, action)` | [`agent_system/environments/env_package/search/envs.py`](../../agent_system/environments/env_package/search/envs.py#L86) |
| `SearchMultiProcessEnv.close` | `def(self)` | [`agent_system/environments/env_package/search/envs.py`](../../agent_system/environments/env_package/search/envs.py#L149) |
| `SearchMultiProcessEnv.reset` | `def(self, kwargs)` | [`agent_system/environments/env_package/search/envs.py`](../../agent_system/environments/env_package/search/envs.py#L99) |
| `SearchMultiProcessEnv.step` | `def(self, actions)` | [`agent_system/environments/env_package/search/envs.py`](../../agent_system/environments/env_package/search/envs.py#L126) |
| `SearchTool.__init__` | `def(self, config, tool_schema)` | [`verl/tools/search_tool.py`](../../verl/tools/search_tool.py#L125) |
| `SearchTool.calc_reward` | `async def(self, instance_id, **kwargs)` | [`verl/tools/search_tool.py`](../../verl/tools/search_tool.py#L253) |
| `SearchTool.create` | `async def(self, instance_id, **kwargs)` | [`verl/tools/search_tool.py`](../../verl/tools/search_tool.py#L176) |
| `SearchTool.execute` | `async def(self, instance_id, parameters, **kwargs)` | [`verl/tools/search_tool.py`](../../verl/tools/search_tool.py#L216) |
| `SearchTool.execute_search` | `def(self, instance_id, query_list, retrieval_service_url, topk, timeout)` | [`verl/tools/search_tool.py`](../../verl/tools/search_tool.py#L193) |
| `SearchTool.get_openai_tool_schema` | `def(self)` | [`verl/tools/search_tool.py`](../../verl/tools/search_tool.py#L172) |
| `SearchTool.release` | `async def(self, instance_id, **kwargs)` | [`verl/tools/search_tool.py`](../../verl/tools/search_tool.py#L256) |
| `SearchToolGroup.__init__` | `def(self, search_url, topk, timeout, log_requests)` | [`agent_system/environments/env_package/search/third_party/skyrl_gym/tools/search.py`](../../agent_system/environments/env_package/search/third_party/skyrl_gym/tools/search.py#L208) |
| `SearchToolGroup._get_shared_session` | `def(cls, base_url)` | [`agent_system/environments/env_package/search/third_party/skyrl_gym/tools/search.py`](../../agent_system/environments/env_package/search/third_party/skyrl_gym/tools/search.py#L190) |
| `SearchToolGroup._query_cache_get` | `def(cls, key)` | [`agent_system/environments/env_package/search/third_party/skyrl_gym/tools/search.py`](../../agent_system/environments/env_package/search/third_party/skyrl_gym/tools/search.py#L174) |
| `SearchToolGroup._query_cache_put` | `def(cls, key, value)` | [`agent_system/environments/env_package/search/third_party/skyrl_gym/tools/search.py`](../../agent_system/environments/env_package/search/third_party/skyrl_gym/tools/search.py#L182) |
| `SearchToolGroup.search` | `def(self, query)` | [`agent_system/environments/env_package/search/third_party/skyrl_gym/tools/search.py`](../../agent_system/environments/env_package/search/third_party/skyrl_gym/tools/search.py#L226) |
| `SeqAllToAll.backward` | `def(ctx, *grad_output)` | [`verl/utils/ulysses.py`](../../verl/utils/ulysses.py#L182) |
| `SeqAllToAll.forward` | `def(ctx, group, local_input, scatter_dim, gather_dim, async_op)` | [`verl/utils/ulysses.py`](../../verl/utils/ulysses.py#L167) |
| `SeqWriter.writeseq` | `def(self, seq)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L26) |
| `set_basic_config` | `def(level)` | [`verl/utils/logging_utils.py`](../../verl/utils/logging_utils.py#L21) |
| `set_expandable_segments` | `def(enable)` | [`verl/utils/device.py`](../../verl/utils/device.py#L89) |
| `set_gamefile` | `def(infos, gamefile)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L37) |
| `set_gamefile` | `def(infos, gamefile)` | [`recipe/hgpo/env_manager.py`](../../recipe/hgpo/env_manager.py#L22) |
| `set_level` | `def(level)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L278) |
| `set_macos_start_method` | `def()` | [`tests/utils/cpu_tests/test_timeout_decorator.py`](../../tests/utils/cpu_tests/test_timeout_decorator.py#L83) |
| `set_pad_token_id` | `def(tokenizer)` | [`verl/utils/tokenizer.py`](../../verl/utils/tokenizer.py#L21) |
| `set_random_seed` | `def(seed)` | [`verl/workers/megatron_workers.py`](../../verl/workers/megatron_workers.py#L52) |
| `set_seed` | `def(seed)` | [`agent_system/environments/env_package/sokoban/sokoban/utils.py`](../../agent_system/environments/env_package/sokoban/sokoban/utils.py#L20) |
| `set_ulysses_sequence_parallel_group` | `def(group)` | [`verl/utils/ulysses.py`](../../verl/utils/ulysses.py#L30) |
| `setup_logger` | `def(session_id, user_log_dir)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/utils.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/utils.py#L30) |
| `setup_logging` | `def(output_dir)` | [`agent_system/environments/env_package/sokoban/sokoban/utils.py`](../../agent_system/environments/env_package/sokoban/sokoban/utils.py#L33) |
| `SFTDataset.__getitem__` | `def(self, item)` | [`verl/utils/dataset/sft_dataset.py`](../../verl/utils/dataset/sft_dataset.py#L118) |
| `SFTDataset.__init__` | `def(self, parquet_files, tokenizer, config)` | [`verl/utils/dataset/sft_dataset.py`](../../verl/utils/dataset/sft_dataset.py#L41) |
| `SFTDataset.__len__` | `def(self)` | [`verl/utils/dataset/sft_dataset.py`](../../verl/utils/dataset/sft_dataset.py#L115) |
| `SFTDataset._download` | `def(self)` | [`verl/utils/dataset/sft_dataset.py`](../../verl/utils/dataset/sft_dataset.py#L72) |
| `SFTDataset._read_files_and_tokenize` | `def(self)` | [`verl/utils/dataset/sft_dataset.py`](../../verl/utils/dataset/sft_dataset.py#L76) |
| `SFTDataset._read_files_and_tokenize.series_to_item` | `def(ls)` | [`verl/utils/dataset/sft_dataset.py`](../../verl/utils/dataset/sft_dataset.py#L77) |
| `SGLangRollout.__init__` | `def(self, actor_module, config, tokenizer, model_hf_config, port, trust_remote_code, device_mesh, **kwargs)` | [`verl/workers/rollout/sglang_rollout/sglang_rollout.py`](../../verl/workers/rollout/sglang_rollout/sglang_rollout.py#L118) |
| `SGLangRollout._async_rollout_a_request` | `async def(self, req, do_sample, is_validate, **kwargs)` | [`verl/workers/rollout/sglang_rollout/sglang_rollout.py`](../../verl/workers/rollout/sglang_rollout/sglang_rollout.py#L622) |
| `SGLangRollout._async_rollout_a_request.calc_reward_and_release_fn` | `async def(name, tool)` | [`verl/workers/rollout/sglang_rollout/sglang_rollout.py`](../../verl/workers/rollout/sglang_rollout/sglang_rollout.py#L738) |
| `SGLangRollout._batch_level_generate_sequences` | `def(self, prompts, **kwargs)` | [`verl/workers/rollout/sglang_rollout/sglang_rollout.py`](../../verl/workers/rollout/sglang_rollout/sglang_rollout.py#L425) |
| `SGLangRollout._handle_engine_call` | `async def(self, _req, do_sample, is_validate, **kwargs)` | [`verl/workers/rollout/sglang_rollout/sglang_rollout.py`](../../verl/workers/rollout/sglang_rollout/sglang_rollout.py#L753) |
| `SGLangRollout._handle_pending_state` | `async def(self, _req)` | [`verl/workers/rollout/sglang_rollout/sglang_rollout.py`](../../verl/workers/rollout/sglang_rollout/sglang_rollout.py#L791) |
| `SGLangRollout._init_distributed_env` | `def(self, device_mesh_cpu, **kwargs)` | [`verl/workers/rollout/sglang_rollout/sglang_rollout.py`](../../verl/workers/rollout/sglang_rollout/sglang_rollout.py#L180) |
| `SGLangRollout._init_inference_engine` | `def(self, trust_remote_code, actor_module, port)` | [`verl/workers/rollout/sglang_rollout/sglang_rollout.py`](../../verl/workers/rollout/sglang_rollout/sglang_rollout.py#L231) |
| `SGLangRollout._init_sampling_params` | `def(self, **kwargs)` | [`verl/workers/rollout/sglang_rollout/sglang_rollout.py`](../../verl/workers/rollout/sglang_rollout/sglang_rollout.py#L289) |
| `SGLangRollout._initialize_tools` | `def(self, config, tokenizer)` | [`verl/workers/rollout/sglang_rollout/sglang_rollout.py`](../../verl/workers/rollout/sglang_rollout/sglang_rollout.py#L304) |
| `SGLangRollout._initialize_tools.initialize_tools_from_config` | `def(tools_config)` | [`verl/workers/rollout/sglang_rollout/sglang_rollout.py`](../../verl/workers/rollout/sglang_rollout/sglang_rollout.py#L338) |
| `SGLangRollout._preprocess_prompt_to_async_rollout_requests` | `def(self, prompts, n)` | [`verl/workers/rollout/sglang_rollout/sglang_rollout.py`](../../verl/workers/rollout/sglang_rollout/sglang_rollout.py#L946) |
| `SGLangRollout._req_level_generate_sequences` | `def(self, prompts, **kwargs)` | [`verl/workers/rollout/sglang_rollout/sglang_rollout.py`](../../verl/workers/rollout/sglang_rollout/sglang_rollout.py#L812) |
| `SGLangRollout._verify_config` | `def(self, model_hf_config)` | [`verl/workers/rollout/sglang_rollout/sglang_rollout.py`](../../verl/workers/rollout/sglang_rollout/sglang_rollout.py#L221) |
| `SGLangRollout.execute_method` | `def(self, method, *args, **kwargs)` | [`verl/workers/rollout/sglang_rollout/sglang_rollout.py`](../../verl/workers/rollout/sglang_rollout/sglang_rollout.py#L1023) |
| `SGLangRollout.generate_sequences` | `def(self, prompts, **kwargs)` | [`verl/workers/rollout/sglang_rollout/sglang_rollout.py`](../../verl/workers/rollout/sglang_rollout/sglang_rollout.py#L418) |
| `SGLangRollout.generate_sequences_with_tools` | `def(self, prompts, **kwargs)` | [`verl/workers/rollout/sglang_rollout/sglang_rollout.py`](../../verl/workers/rollout/sglang_rollout/sglang_rollout.py#L802) |
| `SGLangRollout.offload` | `def(self)` | [`verl/workers/rollout/sglang_rollout/sglang_rollout.py`](../../verl/workers/rollout/sglang_rollout/sglang_rollout.py#L1097) |
| `SGLangRollout.resume` | `def(self)` | [`verl/workers/rollout/sglang_rollout/sglang_rollout.py`](../../verl/workers/rollout/sglang_rollout/sglang_rollout.py#L1089) |
| `SGLangRollout.update_sampling_params` | `def(self, **kwargs)` | [`verl/workers/rollout/sglang_rollout/sglang_rollout.py`](../../verl/workers/rollout/sglang_rollout/sglang_rollout.py#L388) |
| `should_allow_eval` | `def(expr)` | [`verl/utils/reward_score/prime_math/__init__.py`](../../verl/utils/reward_score/prime_math/__init__.py#L200) |
| `SimBrowser.__init__` | `def(self, server)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py#L622) |
| `SimBrowser.click` | `def(self, clickable_name, text_to_clickable)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py#L635) |
| `SimBrowser.get` | `def(self, url, session_id, session_int)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py#L628) |
| `SimBrowser.search` | `def(self, keywords)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py#L646) |
| `SimpleMemory.__getitem__` | `def(self, idx)` | [`agent_system/memory/memory.py`](../../agent_system/memory/memory.py#L31) |
| `SimpleMemory.__init__` | `def(self)` | [`agent_system/memory/memory.py`](../../agent_system/memory/memory.py#L23) |
| `SimpleMemory.__len__` | `def(self)` | [`agent_system/memory/memory.py`](../../agent_system/memory/memory.py#L28) |
| `SimpleMemory.fetch` | `def(self, history_length, obs_key, action_key)` | [`agent_system/memory/memory.py`](../../agent_system/memory/memory.py#L58) |
| `SimpleMemory.reset` | `def(self, batch_size)` | [`agent_system/memory/memory.py`](../../agent_system/memory/memory.py#L34) |
| `SimpleMemory.store` | `def(self, record)` | [`agent_system/memory/memory.py`](../../agent_system/memory/memory.py#L41) |
| `SimServer.__init__` | `def(self, seed, base_url, file_path, attr_path, filter_goals, limit_goals, num_products, human_goals, show_attrs)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py#L291) |
| `SimServer.done` | `def(self, session_id, **kwargs)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py#L484) |
| `SimServer.get_page_name` | `def(self, url)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py#L604) |
| `SimServer.index` | `def(self, session_id, **kwargs)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py#L353) |
| `SimServer.item_page` | `def(self, session_id, **kwargs)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py#L410) |
| `SimServer.item_sub_page` | `def(self, session_id, **kwargs)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py#L453) |
| `SimServer.receive` | `def(self, session_id, current_url, session_int, **kwargs)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py#L519) |
| `SimServer.search_results` | `def(self, session_id, **kwargs)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py#L364) |
| `single_compute_score` | `async def(evaluation_func, completion, reference, task, task_extra_info, executor, timeout)` | [`verl/workers/reward_manager/prime.py`](../../verl/workers/reward_manager/prime.py#L27) |
| `SkillGRPORayTrainer.__init__` | `def(self, **kwargs)` | [`verl/trainer/main_skill_grpo.py`](../../verl/trainer/main_skill_grpo.py#L267) |
| `SkillGRPORayTrainer.fit` | `def(self)` | [`verl/trainer/main_skill_grpo.py`](../../verl/trainer/main_skill_grpo.py#L276) |
| `SkillGRPORayTrainer.fit.dual_validate` | `def()` | [`verl/trainer/main_skill_grpo.py`](../../verl/trainer/main_skill_grpo.py#L284) |
| `SkillGRPORayTrainer.init_workers` | `def(self)` | [`verl/trainer/main_skill_grpo.py`](../../verl/trainer/main_skill_grpo.py#L273) |
| `SkillGRPOTaskRunner.run` | `def(self, config)` | [`verl/trainer/main_skill_grpo.py`](../../verl/trainer/main_skill_grpo.py#L38) |
| `SkillProvider.__init__` | `def(self, skills_dir, skill_all, skills_dirs)` | [`verl/trainer/ppo/rlsd_utils.py`](../../verl/trainer/ppo/rlsd_utils.py#L62) |
| `SkillProvider._build_all_skills_text` | `def(self)` | [`verl/trainer/ppo/rlsd_utils.py`](../../verl/trainer/ppo/rlsd_utils.py#L96) |
| `SkillProvider._get_skill_text` | `def(self, task_type)` | [`verl/trainer/ppo/rlsd_utils.py`](../../verl/trainer/ppo/rlsd_utils.py#L105) |
| `SkillProvider._provider_for_task` | `def(self, task_name)` | [`verl/trainer/ppo/rlsd_utils.py`](../../verl/trainer/ppo/rlsd_utils.py#L90) |
| `SkillProvider.get_privileged_info` | `def(self, gamefile)` | [`verl/trainer/ppo/rlsd_utils.py`](../../verl/trainer/ppo/rlsd_utils.py#L115) |
| `SkillProvider.get_privileged_info_for_sample` | `def(self, task_name, gamefile, data_source, prompt_text)` | [`verl/trainer/ppo/rlsd_utils.py`](../../verl/trainer/ppo/rlsd_utils.py#L187) |
| `SkillProvider.get_privileged_info_from_data_source` | `def(self, data_source, prompt_text)` | [`verl/trainer/ppo/rlsd_utils.py`](../../verl/trainer/ppo/rlsd_utils.py#L156) |
| `SkillProvider.get_privileged_info_from_prompt` | `def(self, prompt_text)` | [`verl/trainer/ppo/rlsd_utils.py`](../../verl/trainer/ppo/rlsd_utils.py#L128) |
| `SkillSDRayTrainer.__init__` | `def(self, *args, **kwargs)` | [`verl/trainer/ppo/skillsd_ray_trainer.py`](../../verl/trainer/ppo/skillsd_ray_trainer.py#L61) |
| `SkillSDRayTrainer._get_sdl_lambda` | `def(self, step)` | [`verl/trainer/ppo/skillsd_ray_trainer.py`](../../verl/trainer/ppo/skillsd_ray_trainer.py#L67) |
| `SkillSDRayTrainer._normalize_task_name` | `def(task_name)` | [`verl/trainer/ppo/skillsd_ray_trainer.py`](../../verl/trainer/ppo/skillsd_ray_trainer.py#L75) |
| `SkillSDRayTrainer._task_kl_loss_coef_tensor` | `def(self, batch)` | [`verl/trainer/ppo/skillsd_ray_trainer.py`](../../verl/trainer/ppo/skillsd_ray_trainer.py#L78) |
| `SkillSDRayTrainer.fit` | `def(self)` | [`verl/trainer/ppo/skillsd_ray_trainer.py`](../../verl/trainer/ppo/skillsd_ray_trainer.py#L106) |
| `SkillSDTaskRunner.run` | `def(self, config)` | [`verl/trainer/main_skillsd.py`](../../verl/trainer/main_skillsd.py#L35) |
| `SkillTrajectoryCollector.__init__` | `def(self, config, tokenizer, processor, skill_provider)` | [`verl/trainer/main_skill_grpo.py`](../../verl/trainer/main_skill_grpo.py#L195) |
| `SkillTrajectoryCollector._inject_skills_into_obs` | `def(self, obs)` | [`verl/trainer/main_skill_grpo.py`](../../verl/trainer/main_skill_grpo.py#L206) |
| `SkillTrajectoryCollector.dynamic_multi_turn_loop` | `def(self, gen_batch, actor_rollout_wg, envs)` | [`verl/trainer/main_skill_grpo.py`](../../verl/trainer/main_skill_grpo.py#L230) |
| `SkillTrajectoryCollector.gather_rollout_data` | `def(self, *args, **kwargs)` | [`verl/trainer/main_skill_grpo.py`](../../verl/trainer/main_skill_grpo.py#L233) |
| `SkillTrajectoryCollector.multi_turn_loop` | `def(self, gen_batch, actor_rollout_wg, envs, is_train)` | [`verl/trainer/main_skill_grpo.py`](../../verl/trainer/main_skill_grpo.py#L236) |
| `SkillTrajectoryCollector.multi_turn_loop.skill_preprocess_batch` | `def(gen_batch, obs, active_mask)` | [`verl/trainer/main_skill_grpo.py`](../../verl/trainer/main_skill_grpo.py#L248) |
| `SkillTrajectoryCollector.preprocess_batch` | `def(self, gen_batch, obs, active_mask)` | [`verl/trainer/main_skill_grpo.py`](../../verl/trainer/main_skill_grpo.py#L223) |
| `SkillTrajectoryCollector.preprocess_single_sample` | `def(self, item, gen_batch, obs)` | [`verl/trainer/main_skill_grpo.py`](../../verl/trainer/main_skill_grpo.py#L220) |
| `SkillTrajectoryCollector.vanilla_multi_turn_loop` | `def(self, gen_batch, actor_rollout_wg, envs)` | [`verl/trainer/main_skill_grpo.py`](../../verl/trainer/main_skill_grpo.py#L227) |
| `skip_if_valid_sandbox` | `def(url)` | [`tests/workers/rollout/test_sglang_async_rollout_sf_tools.py`](../../tests/workers/rollout/test_sglang_async_rollout_sf_tools.py#L130) |
| `skip_if_valid_sandbox.decorator` | `def(func)` | [`tests/workers/rollout/test_sglang_async_rollout_sf_tools.py`](../../tests/workers/rollout/test_sglang_async_rollout_sf_tools.py#L131) |
| `skip_if_valid_sandbox.decorator.wrapper` | `def(*args, **kwargs)` | [`tests/workers/rollout/test_sglang_async_rollout_sf_tools.py`](../../tests/workers/rollout/test_sglang_async_rollout_sf_tools.py#L133) |
| `slice_input_tensor` | `def(x, dim, padding, group)` | [`verl/utils/ulysses.py`](../../verl/utils/ulysses.py#L117) |
| `slow_task` | `def(x)` | [`tests/utils/cpu_tests/test_timeout_decorator.py`](../../tests/utils/cpu_tests/test_timeout_decorator.py#L37) |
| `softmean` | `def(x, beta, dim, keepdim)` | [`recipe/sppo/sppo_ray_trainer.py`](../../recipe/sppo/sppo_ray_trainer.py#L42) |
| `sokoban_projection` | `def(actions)` | [`agent_system/environments/env_package/sokoban/projection.py`](../../agent_system/environments/env_package/sokoban/projection.py#L22) |
| `SokobanEnv.__init__` | `def(self, mode, **kwargs)` | [`agent_system/environments/env_package/sokoban/sokoban/env.py`](../../agent_system/environments/env_package/sokoban/sokoban/env.py#L34) |
| `SokobanEnv.copy` | `def(self)` | [`agent_system/environments/env_package/sokoban/sokoban/env.py`](../../agent_system/environments/env_package/sokoban/sokoban/env.py#L129) |
| `SokobanEnv.finished` | `def(self)` | [`agent_system/environments/env_package/sokoban/sokoban/env.py`](../../agent_system/environments/env_package/sokoban/sokoban/env.py#L81) |
| `SokobanEnv.render` | `def(self, mode)` | [`agent_system/environments/env_package/sokoban/sokoban/env.py`](../../agent_system/environments/env_package/sokoban/sokoban/env.py#L107) |
| `SokobanEnv.reset` | `def(self, seed)` | [`agent_system/environments/env_package/sokoban/sokoban/env.py`](../../agent_system/environments/env_package/sokoban/sokoban/env.py#L53) |
| `SokobanEnv.set_state` | `def(self, rendered_state)` | [`agent_system/environments/env_package/sokoban/sokoban/env.py`](../../agent_system/environments/env_package/sokoban/sokoban/env.py#L153) |
| `SokobanEnv.step` | `def(self, action)` | [`agent_system/environments/env_package/sokoban/sokoban/env.py`](../../agent_system/environments/env_package/sokoban/sokoban/env.py#L87) |
| `SokobanEnv.success` | `def(self)` | [`agent_system/environments/env_package/sokoban/sokoban/env.py`](../../agent_system/environments/env_package/sokoban/sokoban/env.py#L84) |
| `SokobanEnvironmentManager.__init__` | `def(self, envs, projection_f, config)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L254) |
| `SokobanEnvironmentManager.__init__` | `def(self, envs, projection_f, config)` | [`recipe/hgpo/env_manager.py`](../../recipe/hgpo/env_manager.py#L241) |
| `SokobanEnvironmentManager.build_text_obs` | `def(self, infos, text_obs, init)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L309) |
| `SokobanEnvironmentManager.build_text_obs` | `def(self, infos, text_obs, init)` | [`recipe/hgpo/env_manager.py`](../../recipe/hgpo/env_manager.py#L296) |
| `SokobanEnvironmentManager.reset` | `def(self, kwargs)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L259) |
| `SokobanEnvironmentManager.reset` | `def(self, kwargs)` | [`recipe/hgpo/env_manager.py`](../../recipe/hgpo/env_manager.py#L246) |
| `SokobanEnvironmentManager.step` | `def(self, text_actions)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L279) |
| `SokobanEnvironmentManager.step` | `def(self, text_actions)` | [`recipe/hgpo/env_manager.py`](../../recipe/hgpo/env_manager.py#L266) |
| `SokobanMultiProcessEnv.__del__` | `def(self)` | [`agent_system/environments/env_package/sokoban/envs.py`](../../agent_system/environments/env_package/sokoban/envs.py#L173) |
| `SokobanMultiProcessEnv.__init__` | `def(self, seed, env_num, group_n, mode, resources_per_worker, is_train, env_kwargs)` | [`agent_system/environments/env_package/sokoban/envs.py`](../../agent_system/environments/env_package/sokoban/envs.py#L54) |
| `SokobanMultiProcessEnv.close` | `def(self)` | [`agent_system/environments/env_package/sokoban/envs.py`](../../agent_system/environments/env_package/sokoban/envs.py#L165) |
| `SokobanMultiProcessEnv.render` | `def(self, mode, env_idx)` | [`agent_system/environments/env_package/sokoban/envs.py`](../../agent_system/environments/env_package/sokoban/envs.py#L148) |
| `SokobanMultiProcessEnv.reset` | `def(self)` | [`agent_system/environments/env_package/sokoban/envs.py`](../../agent_system/environments/env_package/sokoban/envs.py#L118) |
| `SokobanMultiProcessEnv.step` | `def(self, actions)` | [`agent_system/environments/env_package/sokoban/envs.py`](../../agent_system/environments/env_package/sokoban/envs.py#L91) |
| `SokobanWorker.__init__` | `def(self, mode, env_kwargs)` | [`agent_system/environments/env_package/sokoban/envs.py`](../../agent_system/environments/env_package/sokoban/envs.py#L27) |
| `SokobanWorker.render` | `def(self, mode_for_render)` | [`agent_system/environments/env_package/sokoban/envs.py`](../../agent_system/environments/env_package/sokoban/envs.py#L41) |
| `SokobanWorker.reset` | `def(self, seed_for_reset)` | [`agent_system/environments/env_package/sokoban/envs.py`](../../agent_system/environments/env_package/sokoban/envs.py#L36) |
| `SokobanWorker.step` | `def(self, action)` | [`agent_system/environments/env_package/sokoban/envs.py`](../../agent_system/environments/env_package/sokoban/envs.py#L31) |
| `solve_sokoban` | `def(env, saved_animation_path)` | [`agent_system/environments/env_package/sokoban/sokoban/room_utils.py`](../../agent_system/environments/env_package/sokoban/sokoban/room_utils.py#L134) |
| `sort_placement_group_by_node_ip` | `def(pgs)` | [`verl/single_controller/ray/base.py`](../../verl/single_controller/ray/base.py#L65) |
| `SPINDataParallelPPOActor.compute_log_prob` | `def(self, data)` | [`recipe/spin/dp_actor.py`](../../recipe/spin/dp_actor.py#L33) |
| `SPINDataParallelPPOActor.update_policy_dpo_with_ref` | `def(self, data)` | [`recipe/spin/dp_actor.py`](../../recipe/spin/dp_actor.py#L91) |
| `SPINRolloutRefWorker.compute_log_prob` | `def(self, data)` | [`recipe/spin/fsdp_workers.py`](../../recipe/spin/fsdp_workers.py#L177) |
| `SPINRolloutRefWorker.compute_ref_log_prob` | `def(self, data)` | [`recipe/spin/fsdp_workers.py`](../../recipe/spin/fsdp_workers.py#L150) |
| `SPINRolloutRefWorker.init_model` | `def(self)` | [`recipe/spin/fsdp_workers.py`](../../recipe/spin/fsdp_workers.py#L70) |
| `SPINRolloutRefWorker.update_actor_dpo` | `def(self, data)` | [`recipe/spin/fsdp_workers.py`](../../recipe/spin/fsdp_workers.py#L211) |
| `split_dict_tensor_into_batches` | `def(tensors, batch_size)` | [`verl/utils/torch_functional.py`](../../verl/utils/torch_functional.py#L264) |
| `split_tuple` | `def(expr)` | [`verl/utils/reward_score/prime_math/__init__.py`](../../verl/utils/reward_score/prime_math/__init__.py#L227) |
| `SPPOActorRolloutRefWorker.init_model` | `def(self)` | [`recipe/sppo/sppo_worker.py`](../../recipe/sppo/sppo_worker.py#L40) |
| `squeeze` | `def(x)` | [`verl/utils/model.py`](../../verl/utils/model.py#L45) |
| `step_norm_reward` | `def(step_rewards, response_mask, index, epsilon, remove_std)` | [`gigpo/core_gigpo.py`](../../gigpo/core_gigpo.py#L334) |
| `string_int_check` | `def(val)` | [`verl/utils/reward_score/prime_code/testing_util.py`](../../verl/utils/reward_score/prime_code/testing_util.py#L72) |
| `strip_string` | `def(string)` | [`verl/utils/reward_score/math.py`](../../verl/utils/reward_score/math.py#L162) |
| `stripped_string_compare` | `def(s1, s2)` | [`verl/utils/reward_score/prime_code/testing_util.py`](../../verl/utils/reward_score/prime_code/testing_util.py#L524) |
| `subem_check` | `def(prediction, golden_answers)` | [`verl/utils/reward_score/search_r1_like_qa_em.py`](../../verl/utils/reward_score/search_r1_like_qa_em.py#L53) |
| `sum_hand` | `def(hand)` | [`agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/blackjack.py`](../../agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/blackjack.py#L59) |
| `summarize_group_size` | `def(group_size)` | [`gigpo/core_gigpo.py`](../../gigpo/core_gigpo.py#L49) |
| `symbolic_equal` | `def(a, b, tolerance, timeout)` | [`verl/utils/reward_score/prime_math/grader.py`](../../verl/utils/reward_score/prime_math/grader.py#L282) |
| `symbolic_equal._parse` | `def(s)` | [`verl/utils/reward_score/prime_math/grader.py`](../../verl/utils/reward_score/prime_math/grader.py#L283) |
| `sync_model_parameters_global` | `def(layer)` | [`tests/models/test_transformers_ulysses.py`](../../tests/models/test_transformers_ulysses.py#L66) |
| `sync_reference` | `def(n_traj, max_steps, initial_obs, policy, env_step_sync, rollout_n)` | [`tests/ray_cpu/test_async_rollout_equivalence.py`](../../tests/ray_cpu/test_async_rollout_equivalence.py#L77) |
| `SynchronizedGroupOffloadHandler.__init__` | `def(self, num_offload_group, tensor_need_offloading_checker)` | [`verl/utils/activation_offload.py`](../../verl/utils/activation_offload.py#L136) |
| `SynchronizedGroupOffloadHandler.groupid_reset` | `def(self)` | [`verl/utils/activation_offload.py`](../../verl/utils/activation_offload.py#L144) |
| `SynchronizedGroupOffloadHandler.offload` | `def(src_tensor, pin_memory)` | [`verl/utils/activation_offload.py`](../../verl/utils/activation_offload.py#L166) |
| `SynchronizedGroupOffloadHandler.on_group_commit_backward` | `def(self)` | [`verl/utils/activation_offload.py`](../../verl/utils/activation_offload.py#L160) |
| `SynchronizedGroupOffloadHandler.on_group_commit_forward` | `def(self)` | [`verl/utils/activation_offload.py`](../../verl/utils/activation_offload.py#L154) |
| `SynchronizedGroupOffloadHandler.reload` | `def(state, non_blocking)` | [`verl/utils/activation_offload.py`](../../verl/utils/activation_offload.py#L181) |
| `SynchronizedGroupOffloadHandler.tensor_pop` | `def(self, tensor_tag, **kwargs)` | [`verl/utils/activation_offload.py`](../../verl/utils/activation_offload.py#L203) |
| `SynchronizedGroupOffloadHandler.tensor_push` | `def(self, tensor, **kwargs)` | [`verl/utils/activation_offload.py`](../../verl/utils/activation_offload.py#L188) |
| `tag_visible` | `def(element)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_site_env.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_site_env.py#L212) |
| `tag_visible` | `def(element)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py#L282) |
| `task_raises_value_error` | `def()` | [`tests/utils/cpu_tests/test_timeout_decorator.py`](../../tests/utils/cpu_tests/test_timeout_decorator.py#L44) |
| `task_row_indices` | `def(batch)` | [`verl/trainer/ppo/metric_utils.py`](../../verl/trainer/ppo/metric_utils.py#L94) |
| `TaskBalancedSampler.__init__` | `def(self, dataset, task_balance_config, batch_size, shuffle, seed)` | [`verl/trainer/main_ppo.py`](../../verl/trainer/main_ppo.py#L229) |
| `TaskBalancedSampler.__iter__` | `def(self)` | [`verl/trainer/main_ppo.py`](../../verl/trainer/main_ppo.py#L281) |
| `TaskBalancedSampler.__len__` | `def(self)` | [`verl/trainer/main_ppo.py`](../../verl/trainer/main_ppo.py#L305) |
| `TaskBalancedSampler._indices_for_required_size` | `def(self, indices, required, rng)` | [`verl/trainer/main_ppo.py`](../../verl/trainer/main_ppo.py#L271) |
| `TaskRunner.run` | `def(self, config)` | [`recipe/dapo/main_dapo.py`](../../recipe/dapo/main_dapo.py#L48) |
| `TaskRunner.run` | `def(self, config)` | [`recipe/hgpo/main_hgpo.py`](../../recipe/hgpo/main_hgpo.py#L55) |
| `TaskRunner.run` | `def(self, config)` | [`recipe/spin/main_spin.py`](../../recipe/spin/main_spin.py#L44) |
| `TaskRunner.run` | `def(self, config)` | [`recipe/sppo/main_sppo.py`](../../recipe/sppo/main_sppo.py#L53) |
| `TaskRunner.run` | `def(self, config)` | [`verl/trainer/main_ppo.py`](../../verl/trainer/main_ppo.py#L56) |
| `TensorBoardOutputFormat.__init__` | `def(self, dir)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L162) |
| `TensorBoardOutputFormat.close` | `def(self)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L189) |
| `TensorBoardOutputFormat.writekvs` | `def(self, kvs)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L177) |
| `TensorBoardOutputFormat.writekvs.summary_val` | `def(k, v)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L178) |
| `test` | `def()` | [`tests/ray_cpu/test_check_worker_alive.py`](../../tests/ray_cpu/test_check_worker_alive.py#L20) |
| `test` | `def()` | [`tests/ray_gpu/test_driverfunc_to_worker.py`](../../tests/ray_gpu/test_driverfunc_to_worker.py#L50) |
| `test` | `def()` | [`tests/ray_gpu/test_high_level_scheduling_api.py`](../../tests/ray_gpu/test_high_level_scheduling_api.py#L33) |
| `test_activation_offloading` | `def(world_size, strategy, tmp_path)` | [`tests/utils/gpu_tests/test_activation_offload.py`](../../tests/utils/gpu_tests/test_activation_offload.py#L134) |
| `test_all_gather_data_proto` | `def()` | [`tests/distributed/test_tensor_dict.py`](../../tests/distributed/test_tensor_dict.py#L27) |
| `test_all_gather_torch` | `def()` | [`tests/ray_gpu/test_worker_group_torch.py`](../../tests/ray_gpu/test_worker_group_torch.py#L63) |
| `test_all_gather_torch_v2` | `def()` | [`tests/ray_gpu/test_worker_group_torch.py`](../../tests/ray_gpu/test_worker_group_torch.py#L87) |
| `test_always_recopy_flag` | `def(tmp_path, monkeypatch)` | [`tests/utils/cpu_tests/test_fs.py`](../../tests/utils/cpu_tests/test_fs.py#L66) |
| `test_always_recopy_flag.fake_copy` | `def(src, dst, *args, **kwargs)` | [`tests/utils/cpu_tests/test_fs.py`](../../tests/utils/cpu_tests/test_fs.py#L72) |
| `test_async_sglang_rollout_w_tool` | `def()` | [`tests/workers/rollout/test_sglang_async_rollout_w_tools.py`](../../tests/workers/rollout/test_sglang_async_rollout_w_tools.py#L42) |
| `test_auto_padding` | `def()` | [`tests/ray_cpu/test_auto_padding.py`](../../tests/ray_cpu/test_auto_padding.py#L40) |
| `test_basics` | `def()` | [`tests/ray_cpu/test_ray_local_envs.py`](../../tests/ray_cpu/test_ray_local_envs.py#L36) |
| `test_basics` | `def()` | [`tests/ray_gpu/test_worker_group_basics.py`](../../tests/ray_gpu/test_worker_group_basics.py#L90) |
| `test_check_correctness` | `def()` | [`tests/sandbox/test_sandbox.py`](../../tests/sandbox/test_sandbox.py#L167) |
| `test_chunk_concat` | `def()` | [`tests/test_protocol.py`](../../tests/test_protocol.py#L105) |
| `test_colocated_workers` | `def()` | [`tests/ray_gpu/test_colocated_workers.py`](../../tests/ray_gpu/test_colocated_workers.py#L51) |
| `test_colocated_workers_fused` | `def()` | [`tests/ray_gpu/test_colocated_workers_fused.py`](../../tests/ray_gpu/test_colocated_workers_fused.py#L51) |
| `test_committed_expectations_files_self_consistent` | `def()` | [`tests/trainer/test_expected_config.py`](../../tests/trainer/test_expected_config.py#L108) |
| `test_compact_record_equivalence` | `def()` | [`tests/ray_cpu/test_rollout_speedup_mechanisms.py`](../../tests/ray_cpu/test_rollout_speedup_mechanisms.py#L298) |
| `test_compact_record_equivalence.simulate` | `def(compact)` | [`tests/ray_cpu/test_rollout_speedup_mechanisms.py`](../../tests/ray_cpu/test_rollout_speedup_mechanisms.py#L302) |
| `test_configs` | `def()` | [`tests/models/test_transformers_ulysses.py`](../../tests/models/test_transformers_ulysses.py#L48) |
| `test_connection` | `def(name, url, timeout)` | [`scripts/diagnose.py`](../../scripts/diagnose.py#L50) |
| `test_continuous_batching_overlap` | `def()` | [`tests/ray_cpu/test_async_rollout_equivalence.py`](../../tests/ray_cpu/test_async_rollout_equivalence.py#L224) |
| `test_continuous_score_consistency` | `def()` | [`tests/sandbox/test_sandbox.py`](../../tests/sandbox/test_sandbox.py#L143) |
| `test_conversion` | `def(megatron_model_provider, tfconfig, output_path, model)` | [`scripts/converter_hf_to_mcore.py`](../../scripts/converter_hf_to_mcore.py#L58) |
| `test_copy_from_hdfs_with_mocks` | `def(tmp_path, monkeypatch)` | [`tests/utils/cpu_tests/test_fs.py`](../../tests/utils/cpu_tests/test_fs.py#L43) |
| `test_copy_from_hdfs_with_mocks.fake_copy` | `def(src, dst, *args, **kwargs)` | [`tests/utils/cpu_tests/test_fs.py`](../../tests/utils/cpu_tests/test_fs.py#L48) |
| `test_data_transfer` | `def()` | [`tests/ray_gpu/test_data_transfer.py`](../../tests/ray_gpu/test_data_transfer.py#L46) |
| `test_dataproto_fold_unfold` | `def()` | [`tests/test_protocol.py`](../../tests/test_protocol.py#L219) |
| `test_dataproto_index` | `def()` | [`tests/test_protocol.py`](../../tests/test_protocol.py#L278) |
| `test_dataproto_no_batch` | `def()` | [`tests/test_protocol.py`](../../tests/test_protocol.py#L379) |
| `test_dataproto_pad_unpad` | `def()` | [`tests/test_protocol.py`](../../tests/test_protocol.py#L167) |
| `test_dataproto_unfold_column_chunks` | `def()` | [`tests/test_protocol.py`](../../tests/test_protocol.py#L414) |
| `test_decorator_async_function` | `def(ray_init_shutdown)` | [`tests/ray_cpu/test_decorator.py`](../../tests/ray_cpu/test_decorator.py#L99) |
| `test_decorator_dp_compute` | `def(ray_init_shutdown)` | [`tests/ray_cpu/test_decorator.py`](../../tests/ray_cpu/test_decorator.py#L65) |
| `test_distributed_masked_mean` | `def(world_size, tmp_path)` | [`tests/utils/gpu_tests/test_torch_functional.py`](../../tests/utils/gpu_tests/test_torch_functional.py#L94) |
| `test_distributed_mean_max_min_std` | `def(world_size, tmp_path)` | [`tests/utils/gpu_tests/test_torch_functional.py`](../../tests/utils/gpu_tests/test_torch_functional.py#L56) |
| `test_empty_task_slice_is_skipped` | `def()` | [`tests/trainer/test_opd_routing.py`](../../tests/trainer/test_opd_routing.py#L88) |
| `test_env_kwargs_equal` | `def()` | [`tests/ray_cpu/test_rollout_speedup_mechanisms.py`](../../tests/ray_cpu/test_rollout_speedup_mechanisms.py#L282) |
| `test_env_reset_prefetch` | `def()` | [`tests/ray_cpu/test_rollout_speedup_mechanisms.py`](../../tests/ray_cpu/test_rollout_speedup_mechanisms.py#L239) |
| `test_equivalence_all_finish_immediately` | `def()` | [`tests/ray_cpu/test_async_rollout_equivalence.py`](../../tests/ray_cpu/test_async_rollout_equivalence.py#L208) |
| `test_equivalence_capped_at_max_steps` | `def()` | [`tests/ray_cpu/test_async_rollout_equivalence.py`](../../tests/ray_cpu/test_async_rollout_equivalence.py#L199) |
| `test_equivalence_mixed_horizons` | `def()` | [`tests/ray_cpu/test_async_rollout_equivalence.py`](../../tests/ray_cpu/test_async_rollout_equivalence.py#L187) |
| `test_flash_attn_cross_entropy` | `def()` | [`tests/gpu_utility/test_ops.py`](../../tests/gpu_utility/test_ops.py#L16) |
| `test_flops_counter` | `def(config_type)` | [`tests/utils/gpu_tests/test_flops_counter.py`](../../tests/utils/gpu_tests/test_flops_counter.py#L131) |
| `test_fn_name_success_single_case` | `def()` | [`tests/reward_score/test_sandbox_fusion.py`](../../tests/reward_score/test_sandbox_fusion.py#L547) |
| `test_fsdp_ckpt` | `def(strategy)` | [`tests/utils/gpu_tests/checkpoint/test_fsdp_ckpt.py`](../../tests/utils/gpu_tests/checkpoint/test_fsdp_ckpt.py#L30) |
| `test_function` | `def()` | [`tests/utils/cpu_tests/test_module.py`](../../tests/utils/cpu_tests/test_module.py#L30) |
| `test_fused_workers` | `def()` | [`tests/ray_cpu/test_fused_workers.py`](../../tests/ray_cpu/test_fused_workers.py#L58) |
| `test_hf_casual_fwd_bwd` | `def(test_config)` | [`tests/models/test_transformers_ulysses.py`](../../tests/models/test_transformers_ulysses.py#L73) |
| `test_hf_casual_models` | `def()` | [`tests/models/test_transformer.py`](../../tests/models/test_transformer.py#L39) |
| `test_hf_rollout` | `def(n, do_sample, validate)` | [`tests/workers/rollout/test_hf_rollout.py`](../../tests/workers/rollout/test_hf_rollout.py#L95) |
| `test_hf_value_models` | `def()` | [`tests/models/test_transformer.py`](../../tests/models/test_transformer.py#L97) |
| `test_image_rl_data` | `def()` | [`tests/utils/gpu_tests/dataset/test_rl_dataset.py`](../../tests/utils/gpu_tests/dataset/test_rl_dataset.py#L69) |
| `test_import` | `def()` | [`tests/sanity/test_import.py`](../../tests/sanity/test_import.py#L16) |
| `test_in_thread_timeout` | `def()` | [`tests/utils/cpu_tests/test_timeout_decorator.py`](../../tests/utils/cpu_tests/test_timeout_decorator.py#L200) |
| `test_in_thread_timeout.slow_task_in_thread` | `def()` | [`tests/utils/cpu_tests/test_timeout_decorator.py`](../../tests/utils/cpu_tests/test_timeout_decorator.py#L205) |
| `test_in_thread_timeout.thread_target` | `def()` | [`tests/utils/cpu_tests/test_timeout_decorator.py`](../../tests/utils/cpu_tests/test_timeout_decorator.py#L217) |
| `test_integration_compile_error` | `def()` | [`tests/reward_score/test_sandbox_fusion.py`](../../tests/reward_score/test_sandbox_fusion.py#L99) |
| `test_integration_concurrency_all_timeout` | `def()` | [`tests/reward_score/test_sandbox_fusion.py`](../../tests/reward_score/test_sandbox_fusion.py#L495) |
| `test_integration_concurrency_high_load` | `def()` | [`tests/reward_score/test_sandbox_fusion.py`](../../tests/reward_score/test_sandbox_fusion.py#L127) |
| `test_integration_runtime_error` | `def()` | [`tests/reward_score/test_sandbox_fusion.py`](../../tests/reward_score/test_sandbox_fusion.py#L108) |
| `test_integration_runtime_timeout` | `def()` | [`tests/reward_score/test_sandbox_fusion.py`](../../tests/reward_score/test_sandbox_fusion.py#L117) |
| `test_integration_success_correct` | `def()` | [`tests/reward_score/test_sandbox_fusion.py`](../../tests/reward_score/test_sandbox_fusion.py#L78) |
| `test_integration_success_wrong_output` | `def()` | [`tests/reward_score/test_sandbox_fusion.py`](../../tests/reward_score/test_sandbox_fusion.py#L89) |
| `test_internal_exception` | `def()` | [`tests/utils/cpu_tests/test_timeout_decorator.py`](../../tests/utils/cpu_tests/test_timeout_decorator.py#L113) |
| `test_len` | `def()` | [`tests/test_protocol.py`](../../tests/test_protocol.py#L258) |
| `test_load_extern_type_class` | `def()` | [`tests/utils/cpu_tests/test_import_utils.py`](../../tests/utils/cpu_tests/test_import_utils.py#L25) |
| `test_load_extern_type_constant` | `def()` | [`tests/utils/cpu_tests/test_import_utils.py`](../../tests/utils/cpu_tests/test_import_utils.py#L55) |
| `test_load_extern_type_function` | `def()` | [`tests/utils/cpu_tests/test_import_utils.py`](../../tests/utils/cpu_tests/test_import_utils.py#L42) |
| `test_load_extern_type_invalid_module` | `def()` | [`tests/utils/cpu_tests/test_import_utils.py`](../../tests/utils/cpu_tests/test_import_utils.py#L82) |
| `test_load_extern_type_none_path` | `def()` | [`tests/utils/cpu_tests/test_import_utils.py`](../../tests/utils/cpu_tests/test_import_utils.py#L76) |
| `test_load_extern_type_nonexistent_file` | `def()` | [`tests/utils/cpu_tests/test_import_utils.py`](../../tests/utils/cpu_tests/test_import_utils.py#L64) |
| `test_load_extern_type_nonexistent_type` | `def()` | [`tests/utils/cpu_tests/test_import_utils.py`](../../tests/utils/cpu_tests/test_import_utils.py#L70) |
| `test_log_probs_from_logits_response_rmpad` | `def()` | [`tests/gpu_utility/test_torch_functional.py`](../../tests/gpu_utility/test_torch_functional.py#L22) |
| `test_logprobs_from_logits_v2` | `def(dtype)` | [`tests/gpu_utility/test_torch_functional.py`](../../tests/gpu_utility/test_torch_functional.py#L48) |
| `test_lr_scheduler` | `def()` | [`tests/gpu_utility/test_torch_functional.py`](../../tests/gpu_utility/test_torch_functional.py#L67) |
| `test_make_batch_generator_empty` | `def()` | [`tests/utils/gpu_tests/megatron/test_pipeline_parallel.py`](../../tests/utils/gpu_tests/megatron/test_pipeline_parallel.py#L37) |
| `test_make_batch_generator_no_vpp` | `def()` | [`tests/utils/gpu_tests/megatron/test_pipeline_parallel.py`](../../tests/utils/gpu_tests/megatron/test_pipeline_parallel.py#L18) |
| `test_make_batch_generator_with_vpp` | `def()` | [`tests/utils/gpu_tests/megatron/test_pipeline_parallel.py`](../../tests/utils/gpu_tests/megatron/test_pipeline_parallel.py#L25) |
| `test_matching_config_passes` | `def()` | [`tests/trainer/test_expected_config.py`](../../tests/trainer/test_expected_config.py#L52) |
| `test_max_in_flight_cap_preserves_data` | `def()` | [`tests/ray_cpu/test_async_rollout_equivalence.py`](../../tests/ray_cpu/test_async_rollout_equivalence.py#L240) |
| `test_memory_buffers` | `def()` | [`tests/gpu_utility/test_memory_buffers.py`](../../tests/gpu_utility/test_memory_buffers.py#L26) |
| `test_missing_key_fails` | `def()` | [`tests/trainer/test_expected_config.py`](../../tests/trainer/test_expected_config.py#L95) |
| `test_multiprocess_global_concurrency_limit_with_semaphore` | `def()` | [`tests/reward_score/test_sandbox_fusion.py`](../../tests/reward_score/test_sandbox_fusion.py#L388) |
| `test_multiturn_sft_dataset` | `def()` | [`tests/utils/gpu_tests/dataset/test_multiturn_sft_dataset.py`](../../tests/utils/gpu_tests/dataset/test_multiturn_sft_dataset.py#L27) |
| `test_no_grouping` | `def()` | [`tests/ray_cpu/test_async_rollout_equivalence.py`](../../tests/ray_cpu/test_async_rollout_equivalence.py#L216) |
| `test_old_vs_new_from_single_dict` | `def()` | [`tests/test_protocol.py`](../../tests/test_protocol.py#L342) |
| `test_old_vs_new_from_single_dict.OriginProto.from_single_dict` | `def(cls, data, meta_info, auto_padding)` | [`tests/test_protocol.py`](../../tests/test_protocol.py#L352) |
| `test_parallel_put_basic` | `def(init_ray)` | [`tests/ray_cpu/test_ray_utils.py`](../../tests/ray_cpu/test_ray_utils.py#L29) |
| `test_parallel_put_empty` | `def(init_ray)` | [`tests/ray_cpu/test_ray_utils.py`](../../tests/ray_cpu/test_ray_utils.py#L37) |
| `test_parallel_put_workers` | `def(init_ray)` | [`tests/ray_cpu/test_ray_utils.py`](../../tests/ray_cpu/test_ray_utils.py#L43) |
| `test_parallelism` | `def()` | [`tests/sandbox/test_sandbox.py`](../../tests/sandbox/test_sandbox.py#L96) |
| `test_pop` | `def()` | [`tests/test_protocol.py`](../../tests/test_protocol.py#L129) |
| `test_prefetch_merge_all_and_none` | `def()` | [`tests/ray_cpu/test_rollout_speedup_mechanisms.py`](../../tests/ray_cpu/test_rollout_speedup_mechanisms.py#L128) |
| `test_prefetch_merge_duplicated_rows` | `def()` | [`tests/ray_cpu/test_rollout_speedup_mechanisms.py`](../../tests/ray_cpu/test_rollout_speedup_mechanisms.py#L155) |
| `test_prefetch_merge_equivalence` | `def()` | [`tests/ray_cpu/test_rollout_speedup_mechanisms.py`](../../tests/ray_cpu/test_rollout_speedup_mechanisms.py#L100) |
| `test_prime_code` | `def()` | [`tests/sandbox/test_sandbox.py`](../../tests/sandbox/test_sandbox.py#L116) |
| `test_prime_code_sandbox_fusion` | `def()` | [`tests/sandbox/test_sandbox.py`](../../tests/sandbox/test_sandbox.py#L128) |
| `test_prime_math` | `def()` | [`tests/sandbox/test_sandbox.py`](../../tests/sandbox/test_sandbox.py#L175) |
| `test_pure_distillation_loss_equals_teacher_kl` | `def()` | [`tests/trainer/test_opd_routing.py`](../../tests/trainer/test_opd_routing.py#L127) |
| `test_quick_task` | `def()` | [`tests/utils/cpu_tests/test_timeout_decorator.py`](../../tests/utils/cpu_tests/test_timeout_decorator.py#L97) |
| `test_record_and_check_directory_structure` | `def(tmp_path)` | [`tests/utils/cpu_tests/test_fs.py`](../../tests/utils/cpu_tests/test_fs.py#L21) |
| `test_register_new_dispatch_mode` | `def(reset_dispatch_registry)` | [`tests/single_controller/base/test_decorator.py`](../../tests/single_controller/base/test_decorator.py#L31) |
| `test_register_new_dispatch_mode.dummy_collect` | `def(worker_group, output)` | [`tests/single_controller/base/test_decorator.py`](../../tests/single_controller/base/test_decorator.py#L36) |
| `test_register_new_dispatch_mode.dummy_dispatch` | `def(worker_group, *args, **kwargs)` | [`tests/single_controller/base/test_decorator.py`](../../tests/single_controller/base/test_decorator.py#L33) |
| `test_reorder` | `def()` | [`tests/test_protocol.py`](../../tests/test_protocol.py#L94) |
| `test_repeat` | `def()` | [`tests/test_protocol.py`](../../tests/test_protocol.py#L142) |
| `test_rl_dataset` | `def()` | [`tests/utils/gpu_tests/dataset/test_rl_dataset.py`](../../tests/utils/gpu_tests/dataset/test_rl_dataset.py#L29) |
| `test_rm_dataset` | `def()` | [`tests/utils/gpu_tests/dataset/test_rm_dataset.py`](../../tests/utils/gpu_tests/dataset/test_rm_dataset.py#L28) |
| `test_rollout_prefetch_pending_pool` | `def()` | [`tests/ray_cpu/test_rollout_speedup_mechanisms.py`](../../tests/ray_cpu/test_rollout_speedup_mechanisms.py#L180) |
| `test_routing_assigns_correct_teacher_and_restores_order` | `def()` | [`tests/trainer/test_opd_routing.py`](../../tests/trainer/test_opd_routing.py#L71) |
| `test_rvdz` | `def()` | [`tests/ray_gpu/test_rvdz.py`](../../tests/ray_gpu/test_rvdz.py#L37) |
| `test_sample_level_repeat` | `def()` | [`tests/test_protocol.py`](../../tests/test_protocol.py#L389) |
| `test_search_query_cache` | `def()` | [`tests/ray_cpu/test_rollout_speedup_mechanisms.py`](../../tests/ray_cpu/test_rollout_speedup_mechanisms.py#L326) |
| `test_search_query_cache.fake_call_search_api` | `def(**kwargs)` | [`tests/ray_cpu/test_rollout_speedup_mechanisms.py`](../../tests/ray_cpu/test_rollout_speedup_mechanisms.py#L364) |
| `test_seqlen_balancing` | `def()` | [`tests/utils/gpu_tests/test_seqlen_balancing.py`](../../tests/utils/gpu_tests/test_seqlen_balancing.py#L24) |
| `test_seqlen_balancing_distributed_params` | `def(tmp_path)` | [`tests/utils/gpu_tests/test_seqlen_balancing.py`](../../tests/utils/gpu_tests/test_seqlen_balancing.py#L105) |
| `test_sft_cot_dataset` | `def()` | [`tests/utils/gpu_tests/dataset/test_sft_dataset.py`](../../tests/utils/gpu_tests/dataset/test_sft_dataset.py#L27) |
| `test_sft_dataset` | `def()` | [`tests/utils/gpu_tests/dataset/test_sft_dataset.py`](../../tests/utils/gpu_tests/dataset/test_sft_dataset.py#L52) |
| `test_sglang_spmd` | `def()` | [`tests/workers/rollout/test_sglang_spmd.py`](../../tests/workers/rollout/test_sglang_spmd.py#L43) |
| `test_signal_in_thread_does_not_timeout` | `def()` | [`tests/utils/cpu_tests/test_timeout_decorator.py`](../../tests/utils/cpu_tests/test_timeout_decorator.py#L155) |
| `test_signal_in_thread_does_not_timeout.slow_task_in_thread` | `def()` | [`tests/utils/cpu_tests/test_timeout_decorator.py`](../../tests/utils/cpu_tests/test_timeout_decorator.py#L164) |
| `test_signal_in_thread_does_not_timeout.thread_target` | `def()` | [`tests/utils/cpu_tests/test_timeout_decorator.py`](../../tests/utils/cpu_tests/test_timeout_decorator.py#L176) |
| `test_signal_quick_task_main_process` | `def()` | [`tests/utils/cpu_tests/test_timeout_decorator.py`](../../tests/utils/cpu_tests/test_timeout_decorator.py#L127) |
| `test_signal_quick_task_main_process.plain_quick_task_logic` | `def()` | [`tests/utils/cpu_tests/test_timeout_decorator.py`](../../tests/utils/cpu_tests/test_timeout_decorator.py#L131) |
| `test_signal_slow_task_main_process_timeout` | `def()` | [`tests/utils/cpu_tests/test_timeout_decorator.py`](../../tests/utils/cpu_tests/test_timeout_decorator.py#L139) |
| `test_signal_slow_task_main_process_timeout.plain_slow_task_logic` | `def()` | [`tests/utils/cpu_tests/test_timeout_decorator.py`](../../tests/utils/cpu_tests/test_timeout_decorator.py#L143) |
| `test_single_controller_import` | `def()` | [`tests/sanity/test_import.py`](../../tests/sanity/test_import.py#L22) |
| `test_single_mismatch_fails_and_lists_key` | `def()` | [`tests/trainer/test_expected_config.py`](../../tests/trainer/test_expected_config.py#L76) |
| `test_slow_task_timeout` | `def()` | [`tests/utils/cpu_tests/test_timeout_decorator.py`](../../tests/utils/cpu_tests/test_timeout_decorator.py#L104) |
| `test_teacher_kl_gradient_flows_through_student_only` | `def()` | [`tests/trainer/test_opd_routing.py`](../../tests/trainer/test_opd_routing.py#L112) |
| `test_tensor_dict_constructor` | `def()` | [`tests/test_protocol.py`](../../tests/test_protocol.py#L51) |
| `test_tensor_dict_make_iterator` | `def()` | [`tests/test_protocol.py`](../../tests/test_protocol.py#L65) |
| `test_topk_kl_nonnegative_and_student_only_gradient` | `def()` | [`tests/trainer/test_opd_routing.py`](../../tests/trainer/test_opd_routing.py#L168) |
| `test_topk_kl_zero_for_identical_distributions` | `def()` | [`tests/trainer/test_opd_routing.py`](../../tests/trainer/test_opd_routing.py#L157) |
| `test_torch_save_data_proto` | `def()` | [`tests/test_protocol.py`](../../tests/test_protocol.py#L242) |
| `test_trainer_forward_consistency` | `def(trainer, total_steps)` | [`tests/e2e/sft/test_sp_loss_match.py`](../../tests/e2e/sft/test_sp_loss_match.py#L24) |
| `test_union_tensor_dict` | `def()` | [`tests/test_protocol.py`](../../tests/test_protocol.py#L26) |
| `test_unit_api_timeout_error_concurrent` | `def(mock_call_sandbox_api)` | [`tests/reward_score/test_sandbox_fusion.py`](../../tests/reward_score/test_sandbox_fusion.py#L278) |
| `test_unit_api_timeout_error_concurrent.side_effect` | `def(*args, **kwargs)` | [`tests/reward_score/test_sandbox_fusion.py`](../../tests/reward_score/test_sandbox_fusion.py#L287) |
| `test_unit_concurrency_order` | `def(mock_call_sandbox_api)` | [`tests/reward_score/test_sandbox_fusion.py`](../../tests/reward_score/test_sandbox_fusion.py#L243) |
| `test_unit_concurrency_order.side_effect` | `def(*args, **kwargs)` | [`tests/reward_score/test_sandbox_fusion.py`](../../tests/reward_score/test_sandbox_fusion.py#L250) |
| `test_unit_input_output_mismatch` | `def()` | [`tests/reward_score/test_sandbox_fusion.py`](../../tests/reward_score/test_sandbox_fusion.py#L486) |
| `test_unit_invalid_input_format` | `def()` | [`tests/reward_score/test_sandbox_fusion.py`](../../tests/reward_score/test_sandbox_fusion.py#L470) |
| `test_unknown_task_raises` | `def()` | [`tests/trainer/test_opd_routing.py`](../../tests/trainer/test_opd_routing.py#L100) |
| `test_update_existing_dispatch_mode` | `def(reset_dispatch_registry)` | [`tests/single_controller/base/test_decorator.py`](../../tests/single_controller/base/test_decorator.py#L50) |
| `test_update_existing_dispatch_mode.new_collect` | `def(worker_group, output)` | [`tests/single_controller/base/test_decorator.py`](../../tests/single_controller/base/test_decorator.py#L58) |
| `test_update_existing_dispatch_mode.new_dispatch` | `def(worker_group, *args, **kwargs)` | [`tests/single_controller/base/test_decorator.py`](../../tests/single_controller/base/test_decorator.py#L55) |
| `test_update_model_config` | `def(override_kwargs)` | [`tests/utils/cpu_tests/test_model.py`](../../tests/utils/cpu_tests/test_model.py#L30) |
| `test_vllm_multi_turn` | `def(config)` | [`tests/workers/rollout/test_vllm_multi_turn.py`](../../tests/workers/rollout/test_vllm_multi_turn.py#L44) |
| `test_vllm_multi_turn.callback` | `async def(completions, info, exception)` | [`tests/workers/rollout/test_vllm_multi_turn.py`](../../tests/workers/rollout/test_vllm_multi_turn.py#L67) |
| `test_vllm_spmd` | `def()` | [`tests/workers/rollout/test_vllm_spmd.py`](../../tests/workers/rollout/test_vllm_spmd.py#L70) |
| `test_vllm_streaming_response` | `async def(config)` | [`tests/workers/rollout/test_vllm_multi_turn.py`](../../tests/workers/rollout/test_vllm_multi_turn.py#L135) |
| `test_vllm_tool_calling` | `def()` | [`tests/workers/rollout/test_vllm_tool_calling.py`](../../tests/workers/rollout/test_vllm_tool_calling.py#L244) |
| `test_vllm_with_hf` | `def()` | [`tests/workers/rollout/test_vllm_hf_loader.py`](../../tests/workers/rollout/test_vllm_hf_loader.py#L68) |
| `test_vocab_parallel_entropy` | `def()` | [`tests/distributed/test_tensor_dict.py`](../../tests/distributed/test_tensor_dict.py#L58) |
| `TestActor.__init__` | `def(self)` | [`tests/ray_cpu/check_worker_alive/main.py`](../../tests/ray_cpu/check_worker_alive/main.py#L28) |
| `TestActor.__init__` | `def(self)` | [`tests/ray_cpu/test_ray_local_envs.py`](../../tests/ray_cpu/test_ray_local_envs.py#L28) |
| `TestActor.__init__` | `def(self, cuda_visible_devices)` | [`tests/ray_gpu/test_high_level_scheduling_api.py`](../../tests/ray_gpu/test_high_level_scheduling_api.py#L26) |
| `TestActor.__init__` | `def(self, x)` | [`tests/ray_gpu/test_worker_group_basics.py`](../../tests/ray_gpu/test_worker_group_basics.py#L44) |
| `TestActor.__init__` | `def(self, rank, world_size)` | [`tests/workers/rollout/test_sglang_async_rollout_sf_tools.py`](../../tests/workers/rollout/test_sglang_async_rollout_sf_tools.py#L397) |
| `TestActor.foo` | `def(self, wait_time)` | [`tests/ray_cpu/check_worker_alive/main.py`](../../tests/ray_cpu/check_worker_alive/main.py#L32) |
| `TestActor.foo` | `def(self, y)` | [`tests/ray_gpu/test_worker_group_basics.py`](../../tests/ray_gpu/test_worker_group_basics.py#L48) |
| `TestActor.foo_all_to_all` | `def(self, x, y)` | [`tests/ray_gpu/test_worker_group_basics.py`](../../tests/ray_gpu/test_worker_group_basics.py#L60) |
| `TestActor.foo_custom` | `def(self, x, y)` | [`tests/ray_gpu/test_worker_group_basics.py`](../../tests/ray_gpu/test_worker_group_basics.py#L64) |
| `TestActor.foo_one_to_all` | `def(self, x, y)` | [`tests/ray_gpu/test_worker_group_basics.py`](../../tests/ray_gpu/test_worker_group_basics.py#L56) |
| `TestActor.foo_rank_zero` | `def(self, x, y)` | [`tests/ray_gpu/test_worker_group_basics.py`](../../tests/ray_gpu/test_worker_group_basics.py#L52) |
| `TestActor.get_node_id` | `def(self)` | [`tests/ray_gpu/test_high_level_scheduling_api.py`](../../tests/ray_gpu/test_high_level_scheduling_api.py#L29) |
| `TestActor.get_rank` | `def(self)` | [`tests/workers/rollout/test_sglang_async_rollout_sf_tools.py`](../../tests/workers/rollout/test_sglang_async_rollout_sf_tools.py#L406) |
| `TestActor.get_time` | `def(self, timeout)` | [`tests/workers/rollout/test_sglang_async_rollout_sf_tools.py`](../../tests/workers/rollout/test_sglang_async_rollout_sf_tools.py#L415) |
| `TestActor.getenv` | `def(self, key)` | [`tests/ray_cpu/test_ray_local_envs.py`](../../tests/ray_cpu/test_ray_local_envs.py#L31) |
| `TestActor.ping` | `def(self)` | [`tests/workers/rollout/test_sglang_async_rollout_sf_tools.py`](../../tests/workers/rollout/test_sglang_async_rollout_sf_tools.py#L409) |
| `TestActor.record_execution_time` | `def(self, time)` | [`tests/workers/rollout/test_sglang_async_rollout_sf_tools.py`](../../tests/workers/rollout/test_sglang_async_rollout_sf_tools.py#L412) |
| `TestActor.record_rank` | `def(self, rank)` | [`tests/workers/rollout/test_sglang_async_rollout_sf_tools.py`](../../tests/workers/rollout/test_sglang_async_rollout_sf_tools.py#L403) |
| `TestActor.verify_rank` | `def(self)` | [`tests/workers/rollout/test_sglang_async_rollout_sf_tools.py`](../../tests/workers/rollout/test_sglang_async_rollout_sf_tools.py#L429) |
| `TestAllGatherActor.__init__` | `def(self, size)` | [`tests/ray_gpu/test_worker_group_torch.py`](../../tests/ray_gpu/test_worker_group_torch.py#L30) |
| `TestAllGatherActor.all_gather` | `def(self)` | [`tests/ray_gpu/test_worker_group_torch.py`](../../tests/ray_gpu/test_worker_group_torch.py#L39) |
| `TestAllGatherActor.init` | `def(self)` | [`tests/ray_gpu/test_worker_group_torch.py`](../../tests/ray_gpu/test_worker_group_torch.py#L34) |
| `TestAllGatherActorV2.__init__` | `def(self, size)` | [`tests/ray_gpu/test_worker_group_torch.py`](../../tests/ray_gpu/test_worker_group_torch.py#L48) |
| `TestAllGatherActorV2.all_gather` | `def(self)` | [`tests/ray_gpu/test_worker_group_torch.py`](../../tests/ray_gpu/test_worker_group_torch.py#L56) |
| `TestAsyncSglangServer.mock_ray_actor` | `def(self)` | [`tests/workers/rollout/test_async_sglang_server.py`](../../tests/workers/rollout/test_async_sglang_server.py#L29) |
| `TestAsyncSglangServer.server_config` | `def(self)` | [`tests/workers/rollout/test_async_sglang_server.py`](../../tests/workers/rollout/test_async_sglang_server.py#L37) |
| `TestAsyncSglangServer.test_init_engine` | `async def(self, mock_start_fastapi_server, mock_list_actors, server_config, mock_ray_actor)` | [`tests/workers/rollout/test_async_sglang_server.py`](../../tests/workers/rollout/test_async_sglang_server.py#L44) |
| `TestBootstrapMetric.test_bootstrap_metric_basic` | `def(self)` | [`tests/trainer/ppo/test_metric_utils.py`](../../tests/trainer/ppo/test_metric_utils.py#L207) |
| `TestBootstrapMetric.test_bootstrap_metric_empty` | `def(self)` | [`tests/trainer/ppo/test_metric_utils.py`](../../tests/trainer/ppo/test_metric_utils.py#L231) |
| `TestCalcMajVal.test_calc_maj_val_basic` | `def(self)` | [`tests/trainer/ppo/test_metric_utils.py`](../../tests/trainer/ppo/test_metric_utils.py#L240) |
| `TestCalcMajVal.test_calc_maj_val_tie` | `def(self)` | [`tests/trainer/ppo/test_metric_utils.py`](../../tests/trainer/ppo/test_metric_utils.py#L253) |
| `TestClass.__init__` | `def(self, value)` | [`tests/utils/cpu_tests/test_module.py`](../../tests/utils/cpu_tests/test_module.py#L20) |
| `TestClass.get_value` | `def(self)` | [`tests/utils/cpu_tests/test_module.py`](../../tests/utils/cpu_tests/test_module.py#L23) |
| `TestComputeDataMetrics.setUp` | `def(self)` | [`tests/trainer/ppo/test_metric_utils.py`](../../tests/trainer/ppo/test_metric_utils.py#L74) |
| `TestComputeDataMetrics.test_compute_data_metrics_with_critic` | `def(self)` | [`tests/trainer/ppo/test_metric_utils.py`](../../tests/trainer/ppo/test_metric_utils.py#L91) |
| `TestComputeDataMetrics.test_compute_data_metrics_without_critic` | `def(self)` | [`tests/trainer/ppo/test_metric_utils.py`](../../tests/trainer/ppo/test_metric_utils.py#L109) |
| `TestComputeDataMetricsByTask.setUp` | `def(self)` | [`tests/trainer/ppo/test_per_task_metrics.py`](../../tests/trainer/ppo/test_per_task_metrics.py#L123) |
| `TestComputeDataMetricsByTask.test_batch_level_success_rate_is_not_split` | `def(self)` | [`tests/trainer/ppo/test_per_task_metrics.py`](../../tests/trainer/ppo/test_per_task_metrics.py#L149) |
| `TestComputeDataMetricsByTask.test_every_overall_metric_has_a_per_task_counterpart` | `def(self)` | [`tests/trainer/ppo/test_per_task_metrics.py`](../../tests/trainer/ppo/test_per_task_metrics.py#L127) |
| `TestComputeDataMetricsByTask.test_no_per_task_metrics_without_task_information` | `def(self)` | [`tests/trainer/ppo/test_per_task_metrics.py`](../../tests/trainer/ppo/test_per_task_metrics.py#L159) |
| `TestComputeDataMetricsByTask.test_per_task_value_matches_the_metric_on_that_task_slice` | `def(self)` | [`tests/trainer/ppo/test_per_task_metrics.py`](../../tests/trainer/ppo/test_per_task_metrics.py#L137) |
| `TestComputeDataMetricsByTask.test_single_task_batch_reports_the_overall_value` | `def(self)` | [`tests/trainer/ppo/test_per_task_metrics.py`](../../tests/trainer/ppo/test_per_task_metrics.py#L153) |
| `TestComputeMetricsByTask.test_metric_fn_sees_only_the_rows_of_one_task` | `def(self)` | [`tests/trainer/ppo/test_per_task_metrics.py`](../../tests/trainer/ppo/test_per_task_metrics.py#L113) |
| `TestComputeMetricsByTask.test_no_per_task_metrics_without_task_information` | `def(self)` | [`tests/trainer/ppo/test_per_task_metrics.py`](../../tests/trainer/ppo/test_per_task_metrics.py#L118) |
| `TestComputeThroughputMetrics.setUp` | `def(self)` | [`tests/trainer/ppo/test_metric_utils.py`](../../tests/trainer/ppo/test_metric_utils.py#L175) |
| `TestComputeThroughputMetrics.test_compute_throughout_metrics` | `def(self)` | [`tests/trainer/ppo/test_metric_utils.py`](../../tests/trainer/ppo/test_metric_utils.py#L183) |
| `TestComputeThroughputMetricsByTask.test_no_per_task_metrics_without_task_information` | `def(self)` | [`tests/trainer/ppo/test_per_task_metrics.py`](../../tests/trainer/ppo/test_per_task_metrics.py#L302) |
| `TestComputeThroughputMetricsByTask.test_per_task_tokens_and_throughput_sum_to_the_overall_value` | `def(self)` | [`tests/trainer/ppo/test_per_task_metrics.py`](../../tests/trainer/ppo/test_per_task_metrics.py#L290) |
| `TestComputeTimingMetrics.setUp` | `def(self)` | [`tests/trainer/ppo/test_metric_utils.py`](../../tests/trainer/ppo/test_metric_utils.py#L126) |
| `TestComputeTimingMetrics.test_compute_timing_metrics` | `def(self, mock_compute_response_info)` | [`tests/trainer/ppo/test_metric_utils.py`](../../tests/trainer/ppo/test_metric_utils.py#L146) |
| `TestIterTaskRowMasks.test_nothing_is_yielded_without_task_information` | `def(self)` | [`tests/trainer/ppo/test_per_task_metrics.py`](../../tests/trainer/ppo/test_per_task_metrics.py#L245) |
| `TestIterTaskRowMasks.test_rows_are_grouped_by_task_id` | `def(self)` | [`tests/trainer/ppo/test_per_task_metrics.py`](../../tests/trainer/ppo/test_per_task_metrics.py#L231) |
| `TestIterTaskRowMasks.test_rows_without_a_task_are_skipped` | `def(self)` | [`tests/trainer/ppo/test_per_task_metrics.py`](../../tests/trainer/ppo/test_per_task_metrics.py#L239) |
| `TestLinearCrossEntropy.__init__` | `def(self, test_case_idx)` | [`tests/kernels/test_linear_cross_entropy.py`](../../tests/kernels/test_linear_cross_entropy.py#L85) |
| `TestLinearCrossEntropy.check_storage` | `def(self, method_name, run_forward)` | [`tests/kernels/test_linear_cross_entropy.py`](../../tests/kernels/test_linear_cross_entropy.py#L225) |
| `TestLinearCrossEntropy.check_storage_all` | `def(self)` | [`tests/kernels/test_linear_cross_entropy.py`](../../tests/kernels/test_linear_cross_entropy.py#L245) |
| `TestLinearCrossEntropy.cleanup` | `def(self)` | [`tests/kernels/test_linear_cross_entropy.py`](../../tests/kernels/test_linear_cross_entropy.py#L88) |
| `TestLinearCrossEntropy.generate_backward_inputs` | `def(self)` | [`tests/kernels/test_linear_cross_entropy.py`](../../tests/kernels/test_linear_cross_entropy.py#L132) |
| `TestLinearCrossEntropy.generate_forward_inputs` | `def(self)` | [`tests/kernels/test_linear_cross_entropy.py`](../../tests/kernels/test_linear_cross_entropy.py#L126) |
| `TestLinearCrossEntropy.generate_hyper` | `def(self)` | [`tests/kernels/test_linear_cross_entropy.py`](../../tests/kernels/test_linear_cross_entropy.py#L96) |
| `TestLinearCrossEntropy.verify_correctness` | `def(self, iterations)` | [`tests/kernels/test_linear_cross_entropy.py`](../../tests/kernels/test_linear_cross_entropy.py#L137) |
| `TestMultiNodeRateLimiterCase.test_rate_limiter` | `def(self)` | [`tests/workers/rollout/test_sglang_async_rollout_sf_tools.py`](../../tests/workers/rollout/test_sglang_async_rollout_sf_tools.py#L538) |
| `TestMultiNodeRateLimiterCase.test_rate_limiter.fn` | `def(i)` | [`tests/workers/rollout/test_sglang_async_rollout_sf_tools.py`](../../tests/workers/rollout/test_sglang_async_rollout_sf_tools.py#L547) |
| `TestMultiNodeRateLimiterCase.world_size` | `def(self)` | [`tests/workers/rollout/test_sglang_async_rollout_sf_tools.py`](../../tests/workers/rollout/test_sglang_async_rollout_sf_tools.py#L535) |
| `TestNormalizeTaskName.test_canonical_tasks_match_by_substring` | `def(self)` | [`tests/trainer/ppo/test_per_task_metrics.py`](../../tests/trainer/ppo/test_per_task_metrics.py#L70) |
| `TestNormalizeTaskName.test_missing_task_is_none` | `def(self)` | [`tests/trainer/ppo/test_per_task_metrics.py`](../../tests/trainer/ppo/test_per_task_metrics.py#L79) |
| `TestNormalizeTaskName.test_unknown_task_keeps_its_own_bucket` | `def(self)` | [`tests/trainer/ppo/test_per_task_metrics.py`](../../tests/trainer/ppo/test_per_task_metrics.py#L76) |
| `TestProcessValidationMetrics.test_process_validation_metrics_basic` | `def(self)` | [`tests/trainer/ppo/test_metric_utils.py`](../../tests/trainer/ppo/test_metric_utils.py#L273) |
| `TestProcessValidationMetrics.test_process_validation_metrics_with_pred` | `def(self)` | [`tests/trainer/ppo/test_metric_utils.py`](../../tests/trainer/ppo/test_metric_utils.py#L298) |
| `TestRayGlobalActorCase.test_basic_multi_process_init` | `def(self)` | [`tests/workers/rollout/test_sglang_async_rollout_sf_tools.py`](../../tests/workers/rollout/test_sglang_async_rollout_sf_tools.py#L453) |
| `TestRayGlobalActorCase.world_size` | `def(self)` | [`tests/workers/rollout/test_sglang_async_rollout_sf_tools.py`](../../tests/workers/rollout/test_sglang_async_rollout_sf_tools.py#L449) |
| `TestReduceMetrics.test_reduce_metrics_basic` | `def(self)` | [`tests/trainer/ppo/test_metric_utils.py`](../../tests/trainer/ppo/test_metric_utils.py#L41) |
| `TestReduceMetrics.test_reduce_metrics_empty` | `def(self)` | [`tests/trainer/ppo/test_metric_utils.py`](../../tests/trainer/ppo/test_metric_utils.py#L52) |
| `TestReduceMetrics.test_reduce_metrics_single_value` | `def(self)` | [`tests/trainer/ppo/test_metric_utils.py`](../../tests/trainer/ppo/test_metric_utils.py#L61) |
| `TestRolloutWithSearchTools.qwen_model_config` | `def(self)` | [`tests/workers/rollout/test_sglang_async_rollout_search_tools.py`](../../tests/workers/rollout/test_sglang_async_rollout_search_tools.py#L94) |
| `TestRolloutWithSearchTools.qwen_tokenizer` | `def(self)` | [`tests/workers/rollout/test_sglang_async_rollout_search_tools.py`](../../tests/workers/rollout/test_sglang_async_rollout_search_tools.py#L86) |
| `TestRolloutWithSearchTools.search_data` | `def(self, qwen_tokenizer)` | [`tests/workers/rollout/test_sglang_async_rollout_search_tools.py`](../../tests/workers/rollout/test_sglang_async_rollout_search_tools.py#L100) |
| `TestRolloutWithSearchTools.search_data_proto` | `def(self, search_data, qwen_tokenizer)` | [`tests/workers/rollout/test_sglang_async_rollout_search_tools.py`](../../tests/workers/rollout/test_sglang_async_rollout_search_tools.py#L118) |
| `TestRolloutWithSearchTools.search_rollout_config` | `def(self)` | [`tests/workers/rollout/test_sglang_async_rollout_search_tools.py`](../../tests/workers/rollout/test_sglang_async_rollout_search_tools.py#L108) |
| `TestRolloutWithSearchTools.test_over_size_case` | `def(self, mock_env, mock_engine, mock_sampling, search_rollout_config, qwen_tokenizer, qwen_model_config, search_data_proto, search_data)` | [`tests/workers/rollout/test_sglang_async_rollout_search_tools.py`](../../tests/workers/rollout/test_sglang_async_rollout_search_tools.py#L192) |
| `TestRolloutWithSearchTools.test_rollout_req_creation` | `def(self, mock_env, mock_engine, mock_sampling, search_rollout_config, qwen_tokenizer, qwen_model_config, search_data_proto)` | [`tests/workers/rollout/test_sglang_async_rollout_search_tools.py`](../../tests/workers/rollout/test_sglang_async_rollout_search_tools.py#L162) |
| `TestRolloutWithSearchTools.test_tool_call_basic_case` | `def(self, mock_sampling, mock_engine, mock_env, mock_execute, search_rollout_config, qwen_tokenizer, qwen_model_config, search_data_proto, search_data)` | [`tests/workers/rollout/test_sglang_async_rollout_search_tools.py`](../../tests/workers/rollout/test_sglang_async_rollout_search_tools.py#L229) |
| `TestRolloutWithSearchTools.test_tool_call_batch_case` | `def(self, mock_sampling, mock_engine, mock_env, mock_execute, search_rollout_config, qwen_tokenizer, qwen_model_config, search_data_proto, search_data)` | [`tests/workers/rollout/test_sglang_async_rollout_search_tools.py`](../../tests/workers/rollout/test_sglang_async_rollout_search_tools.py#L278) |
| `TestRolloutWithSearchTools.test_tool_call_batch_case.hacked_handle_engine_call` | `async def(self, _req, *_args, **_kwargs)` | [`tests/workers/rollout/test_sglang_async_rollout_search_tools.py`](../../tests/workers/rollout/test_sglang_async_rollout_search_tools.py#L325) |
| `TestRolloutWithSearchTools.test_tools_registration` | `def(self, mock_env, mock_engine, mock_sampling, search_rollout_config, qwen_tokenizer, qwen_model_config)` | [`tests/workers/rollout/test_sglang_async_rollout_search_tools.py`](../../tests/workers/rollout/test_sglang_async_rollout_search_tools.py#L149) |
| `TestRolloutWithTools.qwen_model_config` | `def(self)` | [`tests/workers/rollout/test_sglang_async_rollout_sf_tools.py`](../../tests/workers/rollout/test_sglang_async_rollout_sf_tools.py#L152) |
| `TestRolloutWithTools.qwen_tokenizer` | `def(self)` | [`tests/workers/rollout/test_sglang_async_rollout_sf_tools.py`](../../tests/workers/rollout/test_sglang_async_rollout_sf_tools.py#L144) |
| `TestRolloutWithTools.sandbox_data_proto` | `def(self, sandbox_fusion_data, qwen_tokenizer)` | [`tests/workers/rollout/test_sglang_async_rollout_sf_tools.py`](../../tests/workers/rollout/test_sglang_async_rollout_sf_tools.py#L176) |
| `TestRolloutWithTools.sandbox_fusion_data` | `def(self, qwen_tokenizer)` | [`tests/workers/rollout/test_sglang_async_rollout_sf_tools.py`](../../tests/workers/rollout/test_sglang_async_rollout_sf_tools.py#L158) |
| `TestRolloutWithTools.sandbox_fusion_rollout_config` | `def(self)` | [`tests/workers/rollout/test_sglang_async_rollout_sf_tools.py`](../../tests/workers/rollout/test_sglang_async_rollout_sf_tools.py#L166) |
| `TestRolloutWithTools.test_over_size_case` | `def(self, mock_env, mock_engine, mock_sampling, sandbox_fusion_rollout_config, qwen_tokenizer, qwen_model_config, sandbox_data_proto, sandbox_fusion_data)` | [`tests/workers/rollout/test_sglang_async_rollout_sf_tools.py`](../../tests/workers/rollout/test_sglang_async_rollout_sf_tools.py#L248) |
| `TestRolloutWithTools.test_rollout_req_creation` | `def(self, mock_env, mock_engine, mock_sampling, sandbox_fusion_rollout_config, qwen_tokenizer, qwen_model_config, sandbox_data_proto)` | [`tests/workers/rollout/test_sglang_async_rollout_sf_tools.py`](../../tests/workers/rollout/test_sglang_async_rollout_sf_tools.py#L218) |
| `TestRolloutWithTools.test_tool_call_basic_case` | `def(self, mock_env, mock_engine, mock_sampling, sandbox_fusion_rollout_config, qwen_tokenizer, qwen_model_config, sandbox_data_proto, sandbox_fusion_data)` | [`tests/workers/rollout/test_sglang_async_rollout_sf_tools.py`](../../tests/workers/rollout/test_sglang_async_rollout_sf_tools.py#L285) |
| `TestRolloutWithTools.test_tool_call_batch_case` | `def(self, mock_env, mock_engine, mock_sampling, sandbox_fusion_rollout_config, qwen_tokenizer, qwen_model_config, sandbox_data_proto, sandbox_fusion_data)` | [`tests/workers/rollout/test_sglang_async_rollout_sf_tools.py`](../../tests/workers/rollout/test_sglang_async_rollout_sf_tools.py#L329) |
| `TestRolloutWithTools.test_tool_call_batch_case.hacked_handle_engine_call` | `async def(self, _req, do_sample, is_validate, **kwargs)` | [`tests/workers/rollout/test_sglang_async_rollout_sf_tools.py`](../../tests/workers/rollout/test_sglang_async_rollout_sf_tools.py#L354) |
| `TestRolloutWithTools.test_tools_registration` | `def(self, mock_env, mock_engine, mock_sampling, sandbox_fusion_rollout_config, qwen_tokenizer, qwen_model_config)` | [`tests/workers/rollout/test_sglang_async_rollout_sf_tools.py`](../../tests/workers/rollout/test_sglang_async_rollout_sf_tools.py#L206) |
| `TestSingleNodeRateLimiterCase.test_rate_limiter` | `def(self)` | [`tests/workers/rollout/test_sglang_async_rollout_sf_tools.py`](../../tests/workers/rollout/test_sglang_async_rollout_sf_tools.py#L478) |
| `TestSingleNodeRateLimiterCase.test_rate_limiter.fn` | `def(i)` | [`tests/workers/rollout/test_sglang_async_rollout_sf_tools.py`](../../tests/workers/rollout/test_sglang_async_rollout_sf_tools.py#L487) |
| `TestSingleNodeRateLimiterCase.test_rotten_execution` | `def(self)` | [`tests/workers/rollout/test_sglang_async_rollout_sf_tools.py`](../../tests/workers/rollout/test_sglang_async_rollout_sf_tools.py#L508) |
| `TestSingleNodeRateLimiterCase.test_rotten_execution.fn` | `def(i)` | [`tests/workers/rollout/test_sglang_async_rollout_sf_tools.py`](../../tests/workers/rollout/test_sglang_async_rollout_sf_tools.py#L516) |
| `TestSingleNodeRateLimiterCase.world_size` | `def(self)` | [`tests/workers/rollout/test_sglang_async_rollout_sf_tools.py`](../../tests/workers/rollout/test_sglang_async_rollout_sf_tools.py#L475) |
| `TestTaskNameExtraction.test_no_task_information_returns_none` | `def(self)` | [`tests/trainer/ppo/test_per_task_metrics.py`](../../tests/trainer/ppo/test_per_task_metrics.py#L95) |
| `TestTaskNameExtraction.test_task_names_fall_back_to_env_kwargs` | `def(self)` | [`tests/trainer/ppo/test_per_task_metrics.py`](../../tests/trainer/ppo/test_per_task_metrics.py#L88) |
| `TestTaskNameExtraction.test_task_names_from_task_name_field` | `def(self)` | [`tests/trainer/ppo/test_per_task_metrics.py`](../../tests/trainer/ppo/test_per_task_metrics.py#L84) |
| `TestTaskNameExtraction.test_task_row_indices_empty_without_task_information` | `def(self)` | [`tests/trainer/ppo/test_per_task_metrics.py`](../../tests/trainer/ppo/test_per_task_metrics.py#L108) |
| `TestTaskNameExtraction.test_task_row_indices_groups_rows` | `def(self)` | [`tests/trainer/ppo/test_per_task_metrics.py`](../../tests/trainer/ppo/test_per_task_metrics.py#L99) |
| `TestTrainerLevelPerTaskMetrics.test_entropy_loss_is_reported_per_task` | `def(self)` | [`tests/trainer/ppo/test_per_task_metrics.py`](../../tests/trainer/ppo/test_per_task_metrics.py#L196) |
| `TestTrainerLevelPerTaskMetrics.test_invalid_action_ratio_is_reported_per_task` | `def(self)` | [`tests/trainer/ppo/test_per_task_metrics.py`](../../tests/trainer/ppo/test_per_task_metrics.py#L166) |
| `TestTrainerLevelPerTaskMetrics.test_reward_kl_penalty_is_reported_per_task` | `def(self)` | [`tests/trainer/ppo/test_per_task_metrics.py`](../../tests/trainer/ppo/test_per_task_metrics.py#L179) |
| `TestTrainerLevelPerTaskMetrics.test_task_ids_are_attached_for_the_workers` | `def(self)` | [`tests/trainer/ppo/test_per_task_metrics.py`](../../tests/trainer/ppo/test_per_task_metrics.py#L209) |
| `TestTrainerLevelPerTaskMetrics.test_task_ids_are_not_attached_without_task_information` | `def(self)` | [`tests/trainer/ppo/test_per_task_metrics.py`](../../tests/trainer/ppo/test_per_task_metrics.py#L218) |
| `TestTrajectoryResponseTokens.setUp` | `def(self)` | [`tests/trainer/ppo/test_per_task_metrics.py`](../../tests/trainer/ppo/test_per_task_metrics.py#L254) |
| `TestTrajectoryResponseTokens.test_metric_is_over_trajectories_not_turns` | `def(self)` | [`tests/trainer/ppo/test_per_task_metrics.py`](../../tests/trainer/ppo/test_per_task_metrics.py#L265) |
| `TestTrajectoryResponseTokens.test_metric_is_reported_per_task` | `def(self)` | [`tests/trainer/ppo/test_per_task_metrics.py`](../../tests/trainer/ppo/test_per_task_metrics.py#L274) |
| `TestTrajectoryResponseTokens.test_opd_metrics_report_trajectory_tokens_per_task` | `def(self)` | [`tests/trainer/ppo/test_per_task_metrics.py`](../../tests/trainer/ppo/test_per_task_metrics.py#L280) |
| `TestTrajectoryResponseTokens.test_tokens_are_summed_over_the_turns_of_a_trajectory` | `def(self)` | [`tests/trainer/ppo/test_per_task_metrics.py`](../../tests/trainer/ppo/test_per_task_metrics.py#L262) |
| `TestWorker.__init__` | `def(self, rank, world_size, group_name)` | [`tests/ray_gpu/test_rvdz.py`](../../tests/ray_gpu/test_rvdz.py#L20) |
| `TestWorker.init` | `def(self)` | [`tests/ray_gpu/test_rvdz.py`](../../tests/ray_gpu/test_rvdz.py#L26) |
| `TestWorker.test` | `def(self)` | [`tests/ray_gpu/test_rvdz.py`](../../tests/ray_gpu/test_rvdz.py#L31) |
| `text_projection` | `def(text_actions, env_name)` | [`agent_system/environments/env_package/gym_cards/gym-cards/text_wrapper.py`](../../agent_system/environments/env_package/gym_cards/gym-cards/text_wrapper.py#L48) |
| `timeout_limit` | `def(seconds, use_signals)` | [`verl/utils/py_functional.py`](../../verl/utils/py_functional.py#L51) |
| `timeout_limit.decorator` | `def(func)` | [`verl/utils/py_functional.py`](../../verl/utils/py_functional.py#L70) |
| `timeout_limit.decorator.wrapper_mp` | `def(*args, **kwargs)` | [`verl/utils/py_functional.py`](../../verl/utils/py_functional.py#L104) |
| `timeout_limit.decorator.wrapper_signal` | `def(*args, **kwargs)` | [`verl/utils/py_functional.py`](../../verl/utils/py_functional.py#L82) |
| `timeout_limit.decorator.wrapper_signal.handler` | `def(signum, frame)` | [`verl/utils/py_functional.py`](../../verl/utils/py_functional.py#L83) |
| `to_hashable` | `def(x)` | [`gigpo/core_gigpo.py`](../../gigpo/core_gigpo.py#L34) |
| `to_hashable` | `def(x)` | [`recipe/hgpo/core_hgpo.py`](../../recipe/hgpo/core_hgpo.py#L13) |
| `to_list_of_dict` | `def(batch)` | [`agent_system/multi_turn_rollout/utils.py`](../../agent_system/multi_turn_rollout/utils.py#L25) |
| `to_numpy` | `def(data)` | [`agent_system/environments/base.py`](../../agent_system/environments/base.py#L23) |
| `TokenBucketWorker.__init__` | `def(self, rate_limit)` | [`verl/tools/sandbox_fusion_tools.py`](../../verl/tools/sandbox_fusion_tools.py#L45) |
| `TokenBucketWorker.__init__` | `def(self, rate_limit)` | [`verl/tools/search_tool.py`](../../verl/tools/search_tool.py#L51) |
| `TokenBucketWorker.acquire` | `def(self)` | [`verl/tools/sandbox_fusion_tools.py`](../../verl/tools/sandbox_fusion_tools.py#L52) |
| `TokenBucketWorker.acquire` | `def(self)` | [`verl/tools/search_tool.py`](../../verl/tools/search_tool.py#L57) |
| `TokenBucketWorker.get_current_count` | `def(self)` | [`verl/tools/sandbox_fusion_tools.py`](../../verl/tools/sandbox_fusion_tools.py#L61) |
| `TokenBucketWorker.get_current_count` | `def(self)` | [`verl/tools/search_tool.py`](../../verl/tools/search_tool.py#L68) |
| `TokenBucketWorker.release` | `def(self)` | [`verl/tools/sandbox_fusion_tools.py`](../../verl/tools/sandbox_fusion_tools.py#L57) |
| `TokenBucketWorker.release` | `def(self)` | [`verl/tools/search_tool.py`](../../verl/tools/search_tool.py#L63) |
| `tokenize_and_postprocess_data` | `def(prompt, tokenizer, max_length, pad_token_id, left_pad, truncation)` | [`verl/utils/torch_functional.py`](../../verl/utils/torch_functional.py#L342) |
| `ToolChatCompletionScheduler.__init__` | `def(self, config, model_path, server_addresses, sandbox_address, system_prompt, **kwargs)` | [`tests/workers/rollout/test_vllm_tool_calling.py`](../../tests/workers/rollout/test_vllm_tool_calling.py#L109) |
| `ToolChatCompletionScheduler.generate_sequences` | `async def(self, batch, **sampling_params)` | [`tests/workers/rollout/test_vllm_tool_calling.py`](../../tests/workers/rollout/test_vllm_tool_calling.py#L126) |
| `ToolChatCompletionScheduler.generate_sequences.callback` | `async def(completions, info, exception)` | [`tests/workers/rollout/test_vllm_tool_calling.py`](../../tests/workers/rollout/test_vllm_tool_calling.py#L149) |
| `ToolChatCompletionScheduler.sandbox_code_execution` | `async def(self, code)` | [`tests/workers/rollout/test_vllm_tool_calling.py`](../../tests/workers/rollout/test_vllm_tool_calling.py#L114) |
| `top_level_decorated_quick_task_signal` | `def()` | [`tests/utils/cpu_tests/test_timeout_decorator.py`](../../tests/utils/cpu_tests/test_timeout_decorator.py#L52) |
| `top_level_decorated_slow_task_signal` | `def()` | [`tests/utils/cpu_tests/test_timeout_decorator.py`](../../tests/utils/cpu_tests/test_timeout_decorator.py#L62) |
| `topk_kl_per_token` | `def(student_topk_logprob, teacher_topk_logprob, eps)` | [`verl/trainer/ppo/core_algos.py`](../../verl/trainer/ppo/core_algos.py#L677) |
| `torch_to_numpy` | `def(tensor, is_object)` | [`agent_system/multi_turn_rollout/utils.py`](../../agent_system/multi_turn_rollout/utils.py#L40) |
| `Tracking.__del__` | `def(self)` | [`verl/utils/tracking.py`](../../verl/utils/tracking.py#L132) |
| `Tracking.__init__` | `def(self, project_name, experiment_name, default_backend, config)` | [`verl/utils/tracking.py`](../../verl/utils/tracking.py#L38) |
| `Tracking.log` | `def(self, data, step, backend)` | [`verl/utils/tracking.py`](../../verl/utils/tracking.py#L127) |
| `train` | `def(agent, eval_env, test_env, envs, args)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/train_rl.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/train_rl.py#L74) |
| `Trainer.__init__` | `def(self)` | [`tests/ray_gpu/detached_worker/server.py`](../../tests/ray_gpu/detached_worker/server.py#L46) |
| `Trainer.init_model` | `def(self)` | [`tests/ray_gpu/detached_worker/server.py`](../../tests/ray_gpu/detached_worker/server.py#L68) |
| `Trainer.init_model.megatron_actor_model_provider` | `def(pre_process, post_process)` | [`tests/ray_gpu/detached_worker/server.py`](../../tests/ray_gpu/detached_worker/server.py#L81) |
| `Trainer.train_model` | `def(self, data)` | [`tests/ray_gpu/detached_worker/server.py`](../../tests/ray_gpu/detached_worker/server.py#L111) |
| `trajectory_advantages` | `def(token_level_rewards, response_mask, index, traj_index, epsilon, remove_std, compute_mean_std_cross_steps)` | [`recipe/hgpo/core_hgpo.py`](../../recipe/hgpo/core_hgpo.py#L101) |
| `TrajectoryCollector.__init__` | `def(self, config, tokenizer, processor)` | [`agent_system/multi_turn_rollout/rollout_loop.py`](../../agent_system/multi_turn_rollout/rollout_loop.py#L211) |
| `TrajectoryCollector._get_preproc_executor` | `def(self)` | [`agent_system/multi_turn_rollout/rollout_loop.py`](../../agent_system/multi_turn_rollout/rollout_loop.py#L239) |
| `TrajectoryCollector._placeholder_single_sample` | `def(self, item, gen_batch, obs, template)` | [`agent_system/multi_turn_rollout/rollout_loop.py`](../../agent_system/multi_turn_rollout/rollout_loop.py#L435) |
| `TrajectoryCollector._prefetch_pending_log_probs` | `def(self, actor_rollout_wg)` | [`agent_system/multi_turn_rollout/rollout_loop.py`](../../agent_system/multi_turn_rollout/rollout_loop.py#L642) |
| `TrajectoryCollector._preprocess_single_sample_threadsafe` | `def(self, item, gen_batch, obs)` | [`agent_system/multi_turn_rollout/rollout_loop.py`](../../agent_system/multi_turn_rollout/rollout_loop.py#L277) |
| `TrajectoryCollector._reset_envs` | `def(self, envs, env_kwargs)` | [`agent_system/multi_turn_rollout/rollout_loop.py`](../../agent_system/multi_turn_rollout/rollout_loop.py#L699) |
| `TrajectoryCollector._run_full_preprocess` | `def(self, items, gen_batch, obs)` | [`agent_system/multi_turn_rollout/rollout_loop.py`](../../agent_system/multi_turn_rollout/rollout_loop.py#L251) |
| `TrajectoryCollector._scatter_active_to_full` | `def(self, active_output, active_idx, batch_size)` | [`agent_system/multi_turn_rollout/rollout_loop.py`](../../agent_system/multi_turn_rollout/rollout_loop.py#L596) |
| `TrajectoryCollector._thread_tokenizer` | `def(self)` | [`agent_system/multi_turn_rollout/rollout_loop.py`](../../agent_system/multi_turn_rollout/rollout_loop.py#L244) |
| `TrajectoryCollector.dynamic_multi_turn_loop` | `def(self, gen_batch, actor_rollout_wg, envs)` | [`agent_system/multi_turn_rollout/rollout_loop.py`](../../agent_system/multi_turn_rollout/rollout_loop.py#L929) |
| `TrajectoryCollector.gather_rollout_data` | `def(self, total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings)` | [`agent_system/multi_turn_rollout/rollout_loop.py`](../../agent_system/multi_turn_rollout/rollout_loop.py#L542) |
| `TrajectoryCollector.multi_turn_loop` | `def(self, gen_batch, actor_rollout_wg, envs, is_train)` | [`agent_system/multi_turn_rollout/rollout_loop.py`](../../agent_system/multi_turn_rollout/rollout_loop.py#L997) |
| `TrajectoryCollector.prefetch_env_reset` | `def(self, envs, env_kwargs)` | [`agent_system/multi_turn_rollout/rollout_loop.py`](../../agent_system/multi_turn_rollout/rollout_loop.py#L678) |
| `TrajectoryCollector.preprocess_batch` | `def(self, gen_batch, obs, active_mask)` | [`agent_system/multi_turn_rollout/rollout_loop.py`](../../agent_system/multi_turn_rollout/rollout_loop.py#L472) |
| `TrajectoryCollector.preprocess_single_sample` | `def(self, item, gen_batch, obs, tokenizer)` | [`agent_system/multi_turn_rollout/rollout_loop.py`](../../agent_system/multi_turn_rollout/rollout_loop.py#L282) |
| `TrajectoryCollector.take_prefetched_log_probs` | `def(self)` | [`agent_system/multi_turn_rollout/rollout_loop.py`](../../agent_system/multi_turn_rollout/rollout_loop.py#L670) |
| `TrajectoryCollector.vanilla_multi_turn_loop` | `def(self, gen_batch, actor_rollout_wg, envs)` | [`agent_system/multi_turn_rollout/rollout_loop.py`](../../agent_system/multi_turn_rollout/rollout_loop.py#L714) |
| `TrajectoryTracker.__init__` | `def(self, hdfs_dir, verbose)` | [`verl/utils/debug/trajectory_tracker.py`](../../verl/utils/debug/trajectory_tracker.py#L52) |
| `TrajectoryTracker.dump` | `def(self, data, name)` | [`verl/utils/debug/trajectory_tracker.py`](../../verl/utils/debug/trajectory_tracker.py#L59) |
| `TrajectoryTracker.wait_for_hdfs` | `def(self)` | [`verl/utils/debug/trajectory_tracker.py`](../../verl/utils/debug/trajectory_tracker.py#L63) |
| `truncatefn` | `def(s, length)` | [`verl/utils/reward_score/prime_code/testing_util.py`](../../verl/utils/reward_score/prime_code/testing_util.py#L39) |
| `two_to_all_dispatch_fn` | `def(worker_group, *args, **kwargs)` | [`tests/ray_gpu/test_worker_group_basics.py`](../../tests/ray_gpu/test_worker_group_basics.py#L26) |
| `ulysses_pad` | `def(input_ids_rmpad, position_ids_rmpad, sp_size)` | [`verl/utils/ulysses.py`](../../verl/utils/ulysses.py#L271) |
| `ulysses_pad_and_slice_inputs` | `def(input_ids_rmpad, position_ids_rmpad, sp_size)` | [`verl/utils/ulysses.py`](../../verl/utils/ulysses.py#L290) |
| `unfold_batch_dim` | `def(data, batch_dims)` | [`verl/protocol.py`](../../verl/protocol.py#L160) |
| `union_numpy_dict` | `def(tensor_dict1, tensor_dict2)` | [`verl/protocol.py`](../../verl/protocol.py#L116) |
| `union_tensor_dict` | `def(tensor_dict1, tensor_dict2)` | [`verl/protocol.py`](../../verl/protocol.py#L104) |
| `union_two_dict` | `def(dict1, dict2)` | [`verl/utils/py_functional.py`](../../verl/utils/py_functional.py#L141) |
| `unpad_dataproto` | `def(data, pad_size)` | [`verl/protocol.py`](../../verl/protocol.py#L98) |
| `unwrap_model` | `def(model, module_instances)` | [`verl/utils/megatron_utils.py`](../../verl/utils/megatron_utils.py#L144) |
| `update_dict_with_config` | `def(dictionary, config)` | [`verl/utils/config.py`](../../verl/utils/config.py#L20) |
| `update_dispatch_mode` | `def(dispatch_mode, dispatch_fn, collect_fn)` | [`verl/single_controller/base/decorator.py`](../../verl/single_controller/base/decorator.py#L457) |
| `update_kwargs_with_config` | `def(dictionary, config)` | [`verl/utils/megatron/tensor_parallel.py`](../../verl/utils/megatron/tensor_parallel.py#L30) |
| `update_model_config` | `def(module_config, override_config_kwargs)` | [`verl/utils/model.py`](../../verl/utils/model.py#L49) |
| `usable_ace` | `def(hand_values)` | [`agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/blackjack.py`](../../agent_system/environments/env_package/gym_cards/gym-cards/gym_cards/envs/blackjack.py#L55) |
| `validate_ulysses_config` | `def(num_heads, ulysses_sequence_size)` | [`verl/utils/ulysses.py`](../../verl/utils/ulysses.py#L318) |
| `ValidationGenerationsLogger.log` | `def(self, loggers, samples, step)` | [`verl/utils/tracking.py`](../../verl/utils/tracking.py#L259) |
| `ValidationGenerationsLogger.log_generation_to_clearml` | `def(self, samples, step)` | [`verl/utils/tracking.py`](../../verl/utils/tracking.py#L341) |
| `ValidationGenerationsLogger.log_generations_to_mlflow` | `def(self, samples, step)` | [`verl/utils/tracking.py`](../../verl/utils/tracking.py#L319) |
| `ValidationGenerationsLogger.log_generations_to_swanlab` | `def(self, samples, step)` | [`verl/utils/tracking.py`](../../verl/utils/tracking.py#L297) |
| `ValidationGenerationsLogger.log_generations_to_wandb` | `def(self, samples, step)` | [`verl/utils/tracking.py`](../../verl/utils/tracking.py#L270) |
| `verify` | `def(solution_str, answer, strict_box_verify, pause_tokens_index)` | [`verl/utils/reward_score/math_dapo.py`](../../verl/utils/reward_score/math_dapo.py#L217) |
| `verify_copy` | `def(src, dest)` | [`verl/utils/fs.py`](../../verl/utils/fs.py#L82) |
| `vLLMAsyncRollout.__init__` | `def(self, *args, **kwargs)` | [`verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`](../../verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py#L405) |
| `vLLMAsyncRollout.execute_method` | `def(self, method, *args, **kwargs)` | [`verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`](../../verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py#L441) |
| `vLLMAsyncRollout.init_worker` | `def(self, all_kwargs)` | [`verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`](../../verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py#L411) |
| `vLLMAsyncRollout.load_model` | `def(self, *args, **kwargs)` | [`verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`](../../verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py#L420) |
| `vLLMAsyncRollout.sleep` | `def(self, *args, **kwargs)` | [`verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`](../../verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py#L427) |
| `vLLMAsyncRollout.wake_up` | `def(self, *args, **kwargs)` | [`verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`](../../verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py#L434) |
| `VLLMHijack.hijack` | `def()` | [`verl/utils/vllm_utils.py`](../../verl/utils/vllm_utils.py#L155) |
| `VLLMHijack.hijack.do_hijack` | `def(target_cls, target_method_name, hooking_method)` | [`verl/utils/vllm_utils.py`](../../verl/utils/vllm_utils.py#L238) |
| `VLLMHijack.hijack.hijack__load_adapter` | `def(self, lora_request)` | [`verl/utils/vllm_utils.py`](../../verl/utils/vllm_utils.py#L156) |
| `vLLMRollout.__init__` | `def(self, actor_module, config, tokenizer, model_hf_config, **kwargs)` | [`verl/workers/rollout/vllm_rollout/vllm_rollout.py`](../../verl/workers/rollout/vllm_rollout/vllm_rollout.py#L68) |
| `vLLMRollout.__init__` | `def(self, model_path, config, tokenizer, model_hf_config, **kwargs)` | [`verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`](../../verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py#L95) |
| `vLLMRollout.generate_sequences` | `def(self, prompts, **kwargs)` | [`verl/workers/rollout/vllm_rollout/vllm_rollout.py`](../../verl/workers/rollout/vllm_rollout/vllm_rollout.py#L185) |
| `vLLMRollout.generate_sequences` | `def(self, prompts, **kwargs)` | [`verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`](../../verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py#L249) |
| `vLLMRollout.update_sampling_params` | `def(self, **kwargs)` | [`verl/workers/rollout/vllm_rollout/vllm_rollout.py`](../../verl/workers/rollout/vllm_rollout/vllm_rollout.py#L168) |
| `vLLMRollout.update_sampling_params` | `def(self, **kwargs)` | [`verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`](../../verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py#L232) |
| `vocab_parallel_entropy` | `def(vocab_parallel_logits)` | [`verl/utils/megatron/tensor_parallel.py`](../../verl/utils/megatron/tensor_parallel.py#L142) |
| `vocab_parallel_log_probs_from_logits` | `def(logits, labels)` | [`verl/utils/megatron/tensor_parallel.py`](../../verl/utils/megatron/tensor_parallel.py#L154) |
| `vocab_parallel_log_probs_from_logits_response_rmpad` | `def(input_ids, attention_mask, logits_rmpad, response_length)` | [`verl/utils/megatron/tensor_parallel.py`](../../verl/utils/megatron/tensor_parallel.py#L161) |
| `WandBOutputFormat.__init__` | `def(self, filename)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L108) |
| `WandBOutputFormat.close` | `def(self)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L117) |
| `WandBOutputFormat.writekvs` | `def(self, kvs)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L114) |
| `warn` | `def(*args)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/logger.py#L270) |
| `WebAgentSiteEnv.__init__` | `def(self, observation_mode, **kwargs)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_site_env.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_site_env.py#L21) |
| `WebAgentSiteEnv._parse_html` | `def(self, html, url)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_site_env.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_site_env.py#L120) |
| `WebAgentSiteEnv.action_space` | `def(self)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_site_env.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_site_env.py#L182) |
| `WebAgentSiteEnv.close` | `def(self)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_site_env.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_site_env.py#L207) |
| `WebAgentSiteEnv.convert_html_to_text` | `def(self, html)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_site_env.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_site_env.py#L149) |
| `WebAgentSiteEnv.get_available_actions` | `def(self)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_site_env.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_site_env.py#L93) |
| `WebAgentSiteEnv.get_instruction_text` | `def(self)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_site_env.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_site_env.py#L143) |
| `WebAgentSiteEnv.get_reward` | `def(self)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_site_env.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_site_env.py#L136) |
| `WebAgentSiteEnv.observation` | `def(self)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_site_env.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_site_env.py#L169) |
| `WebAgentSiteEnv.observation_space` | `def(self)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_site_env.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_site_env.py#L187) |
| `WebAgentSiteEnv.render` | `def(self, mode)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_site_env.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_site_env.py#L203) |
| `WebAgentSiteEnv.reset` | `def(self)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_site_env.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_site_env.py#L190) |
| `WebAgentSiteEnv.state` | `def(self)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_site_env.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_site_env.py#L157) |
| `WebAgentSiteEnv.step` | `def(self, action)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_site_env.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_site_env.py#L50) |
| `WebAgentTextEnv.__init__` | `def(self, observation_mode, file_path, attr_path, server, **kwargs)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py#L36) |
| `WebAgentTextEnv._parse_html` | `def(self, html)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py#L181) |
| `WebAgentTextEnv.close` | `def(self)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py#L278) |
| `WebAgentTextEnv.convert_html_to_text` | `def(self, html, simple)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py#L223) |
| `WebAgentTextEnv.get_available_actions` | `def(self)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py#L140) |
| `WebAgentTextEnv.get_image` | `def(self)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py#L163) |
| `WebAgentTextEnv.get_instruction_text` | `def(self)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py#L175) |
| `WebAgentTextEnv.observation` | `def(self)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py#L195) |
| `WebAgentTextEnv.render` | `def(self, mode)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py#L275) |
| `WebAgentTextEnv.reset` | `def(self, session, instruction_text)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py#L253) |
| `WebAgentTextEnv.state` | `def(self)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py#L212) |
| `WebAgentTextEnv.step` | `def(self, action)` | [`agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py#L97) |
| `WebEnv.__init__` | `def(self, args, split, server, id)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/env.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/env.py#L20) |
| `WebEnv.close` | `def(self)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/env.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/env.py#L237) |
| `WebEnv.estimate_score` | `def(self, atts, opts, verify)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/env.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/env.py#L122) |
| `WebEnv.get_search_texts` | `def(self, atts, query, inst)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/env.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/env.py#L66) |
| `WebEnv.get_valid_actions` | `def(self)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/env.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/env.py#L80) |
| `WebEnv.reset` | `def(self, idx)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/env.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/env.py#L211) |
| `WebEnv.score` | `def(self)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/env.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/env.py#L109) |
| `WebEnv.step` | `def(self, action)` | [`agent_system/environments/env_package/webshop/webshop/baseline_models/env.py`](../../agent_system/environments/env_package/webshop/webshop/baseline_models/env.py#L150) |
| `webshop_projection` | `def(actions)` | [`agent_system/environments/env_package/webshop/projection.py`](../../agent_system/environments/env_package/webshop/projection.py#L19) |
| `WebshopEnvironmentManager.__init__` | `def(self, envs, projection_f, config)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L387) |
| `WebshopEnvironmentManager.__init__` | `def(self, envs, projection_f, config)` | [`recipe/hgpo/env_manager.py`](../../recipe/hgpo/env_manager.py#L374) |
| `WebshopEnvironmentManager._process_batch` | `def(self, batch_idx, total_batch_list, total_infos, success)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L509) |
| `WebshopEnvironmentManager._process_batch` | `def(self, batch_idx, total_batch_list, total_infos, success)` | [`recipe/hgpo/env_manager.py`](../../recipe/hgpo/env_manager.py#L496) |
| `WebshopEnvironmentManager.build_text_obs` | `def(self, text_obs, infos, init)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L465) |
| `WebshopEnvironmentManager.build_text_obs` | `def(self, text_obs, infos, init)` | [`recipe/hgpo/env_manager.py`](../../recipe/hgpo/env_manager.py#L452) |
| `WebshopEnvironmentManager.extract_task` | `def(self, text_obs)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L427) |
| `WebshopEnvironmentManager.extract_task` | `def(self, text_obs)` | [`recipe/hgpo/env_manager.py`](../../recipe/hgpo/env_manager.py#L414) |
| `WebshopEnvironmentManager.format_avail_actions` | `def(self, avail)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L450) |
| `WebshopEnvironmentManager.format_avail_actions` | `def(self, avail)` | [`recipe/hgpo/env_manager.py`](../../recipe/hgpo/env_manager.py#L437) |
| `WebshopEnvironmentManager.format_obs` | `def(self, text_obs)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L435) |
| `WebshopEnvironmentManager.format_obs` | `def(self, text_obs)` | [`recipe/hgpo/env_manager.py`](../../recipe/hgpo/env_manager.py#L422) |
| `WebshopEnvironmentManager.reset` | `def(self, kwargs)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L391) |
| `WebshopEnvironmentManager.reset` | `def(self, kwargs)` | [`recipe/hgpo/env_manager.py`](../../recipe/hgpo/env_manager.py#L378) |
| `WebshopEnvironmentManager.step` | `def(self, text_actions)` | [`agent_system/environments/env_manager.py`](../../agent_system/environments/env_manager.py#L404) |
| `WebshopEnvironmentManager.step` | `def(self, text_actions)` | [`recipe/hgpo/env_manager.py`](../../recipe/hgpo/env_manager.py#L391) |
| `WebshopMultiProcessEnv.__del__` | `def(self)` | [`agent_system/environments/env_package/webshop/envs.py`](../../agent_system/environments/env_package/webshop/envs.py#L253) |
| `WebshopMultiProcessEnv.__init__` | `def(self, seed, env_num, group_n, resources_per_worker, is_train, env_kwargs)` | [`agent_system/environments/env_package/webshop/envs.py`](../../agent_system/environments/env_package/webshop/envs.py#L94) |
| `WebshopMultiProcessEnv.close` | `def(self)` | [`agent_system/environments/env_package/webshop/envs.py`](../../agent_system/environments/env_package/webshop/envs.py#L234) |
| `WebshopMultiProcessEnv.fast_forward` | `def(self, num_resets)` | [`agent_system/environments/env_package/webshop/envs.py`](../../agent_system/environments/env_package/webshop/envs.py#L194) |
| `WebshopMultiProcessEnv.render` | `def(self, mode, env_idx)` | [`agent_system/environments/env_package/webshop/envs.py`](../../agent_system/environments/env_package/webshop/envs.py#L218) |
| `WebshopMultiProcessEnv.reset` | `def(self)` | [`agent_system/environments/env_package/webshop/envs.py`](../../agent_system/environments/env_package/webshop/envs.py#L175) |
| `WebshopMultiProcessEnv.step` | `def(self, actions)` | [`agent_system/environments/env_package/webshop/envs.py`](../../agent_system/environments/env_package/webshop/envs.py#L152) |
| `WebshopWorker.__init__` | `def(self, seed, env_kwargs)` | [`agent_system/environments/env_package/webshop/envs.py`](../../agent_system/environments/env_package/webshop/envs.py#L29) |
| `WebshopWorker.close` | `def(self)` | [`agent_system/environments/env_package/webshop/envs.py`](../../agent_system/environments/env_package/webshop/envs.py#L78) |
| `WebshopWorker.get_available_actions` | `def(self)` | [`agent_system/environments/env_package/webshop/envs.py`](../../agent_system/environments/env_package/webshop/envs.py#L70) |
| `WebshopWorker.get_goals` | `def(self)` | [`agent_system/environments/env_package/webshop/envs.py`](../../agent_system/environments/env_package/webshop/envs.py#L74) |
| `WebshopWorker.render` | `def(self, mode_for_render)` | [`agent_system/environments/env_package/webshop/envs.py`](../../agent_system/environments/env_package/webshop/envs.py#L65) |
| `WebshopWorker.reset` | `def(self, idx)` | [`agent_system/environments/env_package/webshop/envs.py`](../../agent_system/environments/env_package/webshop/envs.py#L57) |
| `WebshopWorker.step` | `def(self, action)` | [`agent_system/environments/env_package/webshop/envs.py`](../../agent_system/environments/env_package/webshop/envs.py#L40) |
| `with_task_suffix` | `def(metrics, task)` | [`verl/trainer/ppo/metric_utils.py`](../../verl/trainer/ppo/metric_utils.py#L128) |
| `Worker.__init__` | `def(self, cuda_visible_devices)` | [`verl/single_controller/base/worker.py`](../../verl/single_controller/base/worker.py#L151) |
| `Worker.__new__` | `def(cls, *args, **kwargs)` | [`verl/single_controller/base/worker.py`](../../verl/single_controller/base/worker.py#L90) |
| `Worker._configure_before_init` | `def(self, register_center_name, rank)` | [`verl/single_controller/base/worker.py`](../../verl/single_controller/base/worker.py#L108) |
| `Worker._configure_with_store` | `def(self, store)` | [`verl/single_controller/base/worker.py`](../../verl/single_controller/base/worker.py#L272) |
| `Worker._setup_env_cuda_visible_devices` | `def(self)` | [`verl/single_controller/base/worker.py`](../../verl/single_controller/base/worker.py#L220) |
| `Worker.env_keys` | `def(cls)` | [`verl/single_controller/base/worker.py`](../../verl/single_controller/base/worker.py#L139) |
| `Worker.execute_func_rank_zero` | `def(self, func, *args, **kwargs)` | [`verl/single_controller/base/worker.py`](../../verl/single_controller/base/worker.py#L323) |
| `Worker.execute_with_func_generator` | `def(self, func, *args, **kwargs)` | [`verl/single_controller/base/worker.py`](../../verl/single_controller/base/worker.py#L308) |
| `Worker.get_cuda_visible_devices` | `def(self)` | [`verl/single_controller/base/worker.py`](../../verl/single_controller/base/worker.py#L290) |
| `Worker.get_fused_worker_by_name` | `def(self, worker_name)` | [`verl/single_controller/base/worker.py`](../../verl/single_controller/base/worker.py#L211) |
| `Worker.get_master_addr_port` | `def(self)` | [`verl/single_controller/base/worker.py`](../../verl/single_controller/base/worker.py#L286) |
| `Worker.rank` | `def(self)` | [`verl/single_controller/base/worker.py`](../../verl/single_controller/base/worker.py#L303) |
| `Worker.world_size` | `def(self)` | [`verl/single_controller/base/worker.py`](../../verl/single_controller/base/worker.py#L298) |
| `WorkerGroup.__init__` | `def(self, resource_pool, **kwargs)` | [`verl/single_controller/base/worker_group.py`](../../verl/single_controller/base/worker_group.py#L129) |
| `WorkerGroup._bind_worker_method` | `def(self, user_defined_cls, func_generator)` | [`verl/single_controller/base/worker_group.py`](../../verl/single_controller/base/worker_group.py#L178) |
| `WorkerGroup._block_until_all_workers_alive` | `def(self)` | [`verl/single_controller/base/worker_group.py`](../../verl/single_controller/base/worker_group.py#L152) |
| `WorkerGroup._is_worker_alive` | `def(self, worker)` | [`verl/single_controller/base/worker_group.py`](../../verl/single_controller/base/worker_group.py#L148) |
| `WorkerGroup.start_worker_aliveness_check` | `def(self, every_n_seconds)` | [`verl/single_controller/base/worker_group.py`](../../verl/single_controller/base/worker_group.py#L161) |
| `WorkerGroup.world_size` | `def(self)` | [`verl/single_controller/base/worker_group.py`](../../verl/single_controller/base/worker_group.py#L174) |
| `WorkerGroupRegisterCenter.__init__` | `def(self, rank_zero_info)` | [`verl/single_controller/base/register_center/ray.py`](../../verl/single_controller/base/register_center/ray.py#L22) |
| `WorkerGroupRegisterCenter.get_rank_zero_info` | `def(self)` | [`verl/single_controller/base/register_center/ray.py`](../../verl/single_controller/base/register_center/ray.py#L27) |
| `WorkerGroupRegisterCenter.get_worker_info` | `def(self)` | [`verl/single_controller/base/register_center/ray.py`](../../verl/single_controller/base/register_center/ray.py#L33) |
| `WorkerGroupRegisterCenter.set_worker_info` | `def(self, rank, node_id)` | [`verl/single_controller/base/register_center/ray.py`](../../verl/single_controller/base/register_center/ray.py#L30) |
| `WorkerHelper._get_free_port` | `def(self)` | [`verl/single_controller/base/worker.py`](../../verl/single_controller/base/worker.py#L67) |
| `WorkerHelper._get_node_ip` | `def(self)` | [`verl/single_controller/base/worker.py`](../../verl/single_controller/base/worker.py#L50) |
| `WorkerHelper._get_node_ip.get_node_ip_by_sdk` | `def()` | [`verl/single_controller/base/worker.py`](../../verl/single_controller/base/worker.py#L51) |
| `WorkerHelper._get_pid` | `def(self)` | [`verl/single_controller/base/worker.py`](../../verl/single_controller/base/worker.py#L75) |
| `WorkerHelper.get_availale_master_addr_port` | `def(self)` | [`verl/single_controller/base/worker.py`](../../verl/single_controller/base/worker.py#L72) |
