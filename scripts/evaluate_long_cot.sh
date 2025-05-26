NUM_GPUS=1

MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B # Base model

MODEL=agentica-org/DeepScaleR-1.5B-Preview # DeepScaleR

MODEL=RUC-AIBOX/STILL-3-1.5B-preview # STILL-3

MODEL=gyr66/spo-tree-666-qwen1.5B-math # Model trained using SPO-tree (2K->4K)

MODEL=gyr66/grpo-qwen1.5B-math # Model trained using GRPO (2K->4K)

OUTPUT_DIR=data/evals/$MODEL

MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=2048,gpu_memory_utilization=0.8,data_parallel_size=$NUM_GPUS,generation_parameters={max_new_tokens:2048,temperature:0.6,top_p:0.95}" # 2K context

MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=4096,gpu_memory_utilization=0.8,data_parallel_size=$NUM_GPUS,generation_parameters={max_new_tokens:4096,temperature:0.6,top_p:0.95}" # 4K context

MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,data_parallel_size=$NUM_GPUS,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}" # 32K context

lighteval vllm $MODEL_ARGS "custom|math_500|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR