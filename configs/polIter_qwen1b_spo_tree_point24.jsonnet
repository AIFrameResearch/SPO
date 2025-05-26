local num_episodes_per_iteration = 1024;
// local num_rollouts_per_sample = 32;  // Here to ensure we sample 16 questions per iteration
local num_dataset_samples_per_iteration = 16;


(import 'polIter_qwen1b_spo_chain_point24.jsonnet')
+ {
  episode_generator+: {
    type: 'hybrid_episode_generator',
    dataset_num_samples_per_iteration: num_dataset_samples_per_iteration,
    // wait_until_memory_release: false, # Seperate to two GPU
    inference_strategy+: {
      type: 'hybrid',
      M: 1333,
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

}
