local num_episodes_per_iteration = 512;
local num_rollouts_per_sample = 8;
local num_dataset_samples_per_iteration = num_episodes_per_iteration / num_rollouts_per_sample;


(import 'polIter_qwen1_5b_base_spo_chain_MATH.jsonnet')
+ {
  episode_generator+: {
    dataset_num_samples_per_iteration: num_dataset_samples_per_iteration,

    type: 'math_episode_generator_w_group_advantages',
    adv_method: 'grpo',
  },
  num_episodes_per_iteration: num_episodes_per_iteration,
  trainer+: {
    general_training_args+: {
      target_train_batch_size: 128,
      per_device_train_batch_size: 2,
      per_device_eval_batch_size: 2,
    },
    params+: {
      use_prob_mask: false,
    },
  },
}
