local tree_expansion_iid = '{{prefix}}{{gen "chain_of_thought" temperature={temperature} top_p={top_p} max_tokens={max_tokens} seed={seed} logprobs={logprobs} save_stop_text="stop_text" stop={stop} n={num_samples}}}';
local tree_question_template = '<｜begin▁of▁sentence｜>You are an intelligent system that calculates 24 point games, put your final calculation expression into \\boxed{{}}, for example \\boxed{{1 * 2 * 3 * 4}}.<｜User｜>{query}<｜Assistant｜><think>\n';

{
  prompt_library+: {
    tree+: {
      expansion+: {
        iid: tree_expansion_iid,
      },
      question_template: tree_question_template,
    },
  },
}
