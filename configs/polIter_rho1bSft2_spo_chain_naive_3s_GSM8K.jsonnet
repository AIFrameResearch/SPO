(import 'polIter_rho1bSft2_spo_chain_GSM8K.jsonnet')
+ {
  episode_generator+: {
    type: 'math_episode_generator_w_mc_advantages_naive',
  },
  trainer+: {
    params+: {
      use_prob_mask: false,
    },
  },
}
