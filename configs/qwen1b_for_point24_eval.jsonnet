local num_samples = 16;
local temperature = 0.35;

local tokenizer = {
  type: 'pretrained',
  hf_model_name: 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
};

local point24_inference_pipeline =
  (import 'prompt_library/qwen_point24.jsonnet')
  + (import 'inference_strategies/cot.jsonnet')
  + {
    inference_strategy+: {
      max_concurrent_programs: 512,
      max_concurrent_generations: 128,

      node_expander+: {
        type: 'efficient_iid',
        program: $.prompt_library.tree.expansion.iid,
        program_kwargs: {
          temperature: temperature,
          top_p: 0.9,
          max_tokens: 2048,
          stop: '"\n\n\nProblem:"',
          logprobs: 0,
        },
        node_text_template: '{chain_of_thought}',

        // Needed to compute max_tokens on the fly
        model_context_size: 4096,
        tokenizer: tokenizer,
      },
      answer_extractor+: {
        type: 'identity',
        node_key_name: 'text',
      },
      samples: num_samples,
      max_depth: 10,

      guidance_llm: (import 'guidance_llms/qwen1b.jsonnet') + { api_base: 'none' },
      no_cache: false,
      question_field: 'query',

      seed: 42,
    },
    task: (import 'tasks/point24.jsonnet'),
    analyzers: [(import 'analyzers/task_performance.jsonnet')],
  };

local point24_validation_inference_pipeline =
  point24_inference_pipeline
  {
    dataset_split: 'validation',
    dataset_portion: 0.01,
    inference_name: 'point24_validation',
  };

{
  inference_pipelines: [
    point24_validation_inference_pipeline,
  ],

  evaluation_vllm_server: {},
}
