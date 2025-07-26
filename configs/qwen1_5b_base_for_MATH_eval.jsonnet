local num_samples = 1;
local temperature = 0;

local tokenizer = {
  type: 'pretrained',
  hf_model_name: 'Qwen/Qwen2.5-1.5B',
};

local math_inference_pipeline =
  (import 'prompt_library/qwen_base.jsonnet')
  + (import 'inference_strategies/tree/iid_expander.jsonnet')
  + (import 'inference_strategies/cot.jsonnet')
  + {
    inference_strategy+: {
      max_concurrent_programs: 512,
      max_concurrent_generations: 128,

      node_expander+: {
        type: 'efficient_iid',
        program_kwargs: {
          temperature: temperature,
          top_p: 0.9,
          max_tokens: 4096,
          stop: '"\n\n\nProblem:"',
          logprobs: 0,
        },
        node_text_template: '{chain_of_thought}',

        // Needed to compute max_tokens on the fly
        model_context_size: 1024,
        tokenizer: tokenizer,
      },
      answer_extractor+: {
        type: 'identity',
        node_key_name: 'text',
      },
      samples: num_samples,
      max_depth: 100,  // not used

      guidance_llm: (import 'guidance_llms/qwen1_5b_base.jsonnet') + { api_base: 'none' },
      no_cache: false,
      question_field: 'query',

      seed: 42,
    },
    task: (import 'tasks/math_inplace_no_answer_prefix.jsonnet'),
    analyzers: [(import 'analyzers/task_performance.jsonnet')],
  };

local math_test_inference_pipeline =
  math_inference_pipeline
  {
    dataset_split: 'test',
    dataset_portion: 1,
    inference_name: 'math_test',
  };

local math_validation_inference_pipeline =
  math_inference_pipeline
  {
    dataset_split: 'validation',
    dataset_portion: 1,
    inference_name: 'math_validation',
  };

local collegeMath_inference_pipeline =
  math_inference_pipeline
  {
    task: (import 'tasks/collegeMath_inplace_no_answer_prefix.jsonnet'),
  };

local collegeMath_test_inference_pipeline =
  collegeMath_inference_pipeline
  {
    dataset_split: 'test',
    dataset_portion: 0.1774308,
    inference_name: 'collegeMath_test',
  };

local olympiadbench_inference_pipeline =
  math_inference_pipeline
  {
    task: (import 'tasks/olympiadbench_inplace_no_answer_prefix.jsonnet'),
  };

local olympiadbench_test_inference_pipeline =
  olympiadbench_inference_pipeline
  {
    dataset_split: 'test',
    dataset_portion: 1,
    inference_name: 'olympiadbench_test',
  };

{
  inference_pipelines: [
    math_test_inference_pipeline,
    // math_validation_inference_pipeline,
    // math_train_inference_pipeline,
    // collegeMath_test_inference_pipeline,
    // olympiadbench_test_inference_pipeline,
  ],

  evaluation_vllm_server: {},
}
