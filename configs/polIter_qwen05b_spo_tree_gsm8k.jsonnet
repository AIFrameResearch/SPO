local num_episodes_per_iteration = 512;
// local num_rollouts_per_sample = 32;  // Here to ensure we sample 16 questions per iteration
local num_dataset_samples_per_iteration = 32;


(import 'polIter_qwen05b_spo_chain_gsm8k.jsonnet')
+ {
  episode_generator+: {
    type: 'hybrid_episode_generator',
    dataset_num_samples_per_iteration: num_dataset_samples_per_iteration,
    // wait_until_memory_release: false, # Seperate to two GPU
    inference_strategy+: {
      type: 'hybrid',
      // M: 1111,
      M: 100,
      max_depth: 3,
      branch_factor_strategy: {
        type: 'list',
        branch_factors: [
          { depth: 0, branch_factor: 6 },
          { depth: 1, branch_factor: 6 },
          { depth: 2, branch_factor: 6 },
        ],
      },
    },
  },
  num_episodes_per_iteration: num_episodes_per_iteration,
  trainer+: {
    num_epochs_per_iteration: 1,
    general_training_args+: {
      target_train_batch_size: 128,
      per_device_train_batch_size: 2,
      per_device_eval_batch_size: 2,
    },
  },
}
