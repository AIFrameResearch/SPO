local hf_model_name = 'realtreetune/deepseekmath-7b-sft-MATH-v2';
local hf_model_revision = '036a8c6189aac6e2fc4e07b46e1e57c6b647bca5';

local query_template = '[MATH_TASK] Problem:\n{problem}\n\nSolution:';
local response_template = '\n{solution}';


(import 'gvar.jsonnet')
+ (import 'prompt_library/generic_MATH_step_by_step.jsonnet')
+ (import 'runtimes/policy_iteration.jsonnet')
+ (import 'trainers/mle.jsonnet')
+ (import 'episode_generators/sft.jsonnet')
+ {
  episode_generator+: {
    query_template: query_template,
    response_template: response_template,
    append_bos_to_query: true,
    append_eos_to_response: true,
    task: (import 'tasks/math_no_answer_prefix.jsonnet'),
  },

  model+: {
    type: 'pretrained_causal_lm',
    hf_model_name: hf_model_name,
    pretrained_args+: {
      revision: hf_model_revision,
    },
  },

  tokenizer+: {
    type: 'pretrained',
    hf_model_name: hf_model_name,
    pretrained_args+: {
      revision: hf_model_revision,
    },
  },

  trainer+: {
    type: 'mle',
    num_epochs_per_iteration: 2,
    training_args+: {
      learning_rate: 5e-5,
      weight_decay: 0.00,
      warmup_ratio: 0.03,

      save_steps: 50,
      checkpoint_keep_steps: 50,

      max_seq_len: 2048,

      // Total batch size for training = 64 (4 GPUs)
      per_device_train_batch_size: 2,
      gradient_accumulation_steps: 8,
      gradient_checkpointing: false,
    },
    loss_reduction_mode: 'per_instance_non_pad_tokens_then_batch',
    deepspeed_config: (import 'deepspeed/zero_2.jsonnet'),
  },
  use_deepspeed: true,

  num_episodes_per_iteration: null,
  num_iterations: 1,
}
+ (import 'sft_deepseekmath_for_MATH_eval.jsonnet')
