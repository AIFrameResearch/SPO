local num_episodes_per_iteration = 1024;
// local num_rollouts_per_sample = 32;
local num_dataset_samples_per_iteration = 16;
local total_num_iterations = 1000;

(import 'polIter_deepseekSft2_vineppo_MATH.jsonnet')
+ {
  episode_generator+: {
    type: 'hybrid_episode_generator',
    dataset_num_samples_per_iteration: num_dataset_samples_per_iteration,
    inference_strategy+: {
      type: 'hybrid',
      M: 66,
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
    num_episodes_per_iteration: num_episodes_per_iteration,

  },
  num_episodes_per_iteration: num_episodes_per_iteration,

  trainer+: {
    num_epochs_per_iteration: 1,
    general_training_args+: {
      target_train_batch_size: 128,
      per_device_train_batch_size: 8,
      per_device_eval_batch_size: 2,
    },
  },
}
