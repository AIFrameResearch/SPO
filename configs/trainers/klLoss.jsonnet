{
  trainer+: {
    params+: {
      kl_penalty_loss_type: 'control_variate',
      kl_penalty_loss_clip_max: 1000000000,
      kl_penalty_loss_clip_min: 0,
    },
  },
}
