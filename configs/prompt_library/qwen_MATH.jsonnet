local tree_expansion_iid = '{{prefix}}{{gen "chain_of_thought" temperature={temperature} top_p={top_p} max_tokens={max_tokens} seed={seed} logprobs={logprobs} save_stop_text="stop_text" stop={stop} n={num_samples}}}';
local tree_question_template = "<｜begin▁of▁sentence｜><｜User｜>Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.\n\n{query}<｜Assistant｜><think>\n";
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
