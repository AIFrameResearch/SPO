(import 'polIter_deepseekSft2_vineppo_MATH.jsonnet')
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
