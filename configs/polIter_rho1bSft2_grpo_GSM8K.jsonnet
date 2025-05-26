(import 'polIter_rho1bSft2_spo_chain_GSM8K.jsonnet')
+ {
  episode_generator+: {
    type: 'math_episode_generator_w_group_advantages',
    adv_method: 'grpo',
  },
  trainer+: {
    params+: {
      use_prob_mask: false,
    },
  },
}
