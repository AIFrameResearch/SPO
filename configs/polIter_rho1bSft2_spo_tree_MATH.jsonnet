local num_episodes_per_iteration = 1024;
// local num_rollouts_per_sample = 32;
local num_dataset_samples_per_iteration = 16;
local total_num_iterations = 1000;

(import 'polIter_rho1bSft2_spo_chain_MATH.jsonnet')
+ {
  episode_generator+: {
    type: 'hybrid_episode_generator',
    dataset_num_samples_per_iteration: num_dataset_samples_per_iteration,
    inference_strategy+: {
      type: 'hybrid',
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
  },
}
