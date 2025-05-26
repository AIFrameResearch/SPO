{
  episode_generator+: {
    replay_buffer_type: 'on_policy',
    only_adv_greater_than_zero: false,
    use_hard_estimation: true,
  },
  trainer+: {
    loss_method: 'dvo',
    params+: {
      init_kl_coef: 0.1,
    },
    general_training_args+: {
      target_train_batch_size: 128,
      per_device_train_batch_size: 8,
      per_device_eval_batch_size: 2,
      // per_device_train_batch_size: 32,
      // per_device_eval_batch_size: 32,
    },
  },
  num_episodes_per_iteration: 1024,
}
