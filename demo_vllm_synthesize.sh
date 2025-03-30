#!/bin/bash

template=msp_template   # template can also be "npc", "knowledge" or "math". Feel free to try others; You can also add your customized data synthesis prompt in code/prompt_templates.py
sample_size=10  # Set sample_size=0 if you want to use the full version of 200k personas.
out_path=vllm_${template}_synthesis_output.jsonl
model_path=meta-llama/Meta-Llama-3-70B-Instruct # feel free to replace it with any other open-sourced LLMs supported by vllm.

# ensure that the necessary libraries such as transformers and vllm are installed or configured properly before running the following command.
PYTHONPATH=. python code/vllm_synthesize.py --model_path meta-llama/Meta-Llama-3-70B-Instruct --template msp_template --sample_size 10 --output_path vllm_msp_synthesis_output.jsonl
