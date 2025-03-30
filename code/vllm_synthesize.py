import argparse
from transformers import AutoTokenizer
import transformers
import torch
import json
from tqdm import tqdm
from vllm import LLM, SamplingParams
from datasets import load_dataset
from code.prompt_templates import msp_template

def request_input_format(user_prompt, tokenizer):
    system_prompt = "You are a helpful assistant."
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text

def main(args):
    # Load the appropriate template

    model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    template = msp_template

    # Load the dataset
    persona_dataset = load_dataset("proj-persona/PersonaHub", data_files="persona.jsonl")['train']
    if args.sample_size > 0:
        persona_dataset = persona_dataset[:args.sample_size]
    print(f"Total number of input personas: {len(persona_dataset['persona'])}")

    # Load the model and tokenizer
    model_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model=model_path, tensor_parallel_size=4) # please set tensor_parallel_size based on the GPUs you are using

    prompts = []
    max_len = 2048

    for persona in persona_dataset['persona']:
        persona = persona.strip()
        user_prompt = template.format(persona=persona)
        prompt = request_input_format(user_prompt, tokenizer)
        prompts.append(prompt)

    print(f"Loaded {len(prompts)} entries to process...\n\n")
    print(f"Sample 0: {prompts[0]}")

    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=max_len, stop=["<|eot_id|>"])
    model_id = "meta-llama/Llama-3.1-8B"

    pipeline = transformers.pipeline(
        "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto",temperature=0.6, top_p=0.95, max_tokens=max_len, stop=["<|eot_id|>"]
    )
    #outputs = llm.generate(prompts, sampling_params)
    outputs=pipeline(prompts)

    with open(args.output_path, 'w') as out:
        for i, output in enumerate(outputs):
            out_txt = output.outputs[0].text
            finish_reason = output.outputs[0].finish_reason
            data = {'prompt': output.prompt, "input persona": persona_dataset['persona'][i].strip(), "finish_reason": finish_reason}
            data['synthesized text'] = out_txt
            out.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"Outputted the results to: {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthesize text using a specified model and template.")
    parser.add_argument('--sample_size', type=int, default=0, help='Number of samples to process from the dataset; Set it to 0 if you want to use the full set of 200k personas.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output file.')
    parser.add_argument(
        '--template', 
        type=str, 
        required=True, 
        choices=['instruction', 'knowledge', 'npc', 'math' , 'msp_template'], 
        help=(
            "Prompt templates. Choose from 'instruction', 'knowledge', 'math' or 'npc'. "
            "You can also add more customized templates in code/templates.py"
        )
    )
    args = parser.parse_args()
    main(args)
