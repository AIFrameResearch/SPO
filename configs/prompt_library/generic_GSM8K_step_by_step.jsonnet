local tree_expansion_iid = '{{prefix}}{{gen "chain_of_thought" temperature={temperature} max_tokens={max_tokens} logprobs={logprobs} save_stop_text="stop_text" stop={stop} n={num_samples} seed={seed}}}';
local tree_question_template = '[MATH_TASK] Problem:
{query}

Solution:';

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
