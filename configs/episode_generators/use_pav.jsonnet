{
  episode_generator+: {
    use_pav: true,
    replay_buffer_type: 'on_policy',
    use_hard_estimation: true,
    only_adv_greater_than_zero: false,
  },
  trainer+: {
    general_training_args+: {
      target_train_batch_size: 128,
      per_device_train_batch_size: 16,
      per_device_eval_batch_size: 16,
    },
  },
  num_episodes_per_iteration: 1024,
}
