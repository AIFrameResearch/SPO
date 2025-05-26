{
  episode_generator+: {
    inference_strategy+: {
      branch_factor_strategy: {
        type: 'list',
        branch_factors: [
          { depth: 0, branch_factor: 8 },
          { depth: 1, branch_factor: 8 },
          { depth: 2, branch_factor: 8 },
        ],
      },
    },
  },
  num_episodes_per_iteration: 1024,
}
