(import 'polIter_rho1bSft2_spo_chain_GSM8K.jsonnet')
+ {
  episode_generator+: {
    type: 'vineppo_episode_generator',
  },
  trainer+: {
    params+: {
      use_prob_mask: false,
    },
  },
}
