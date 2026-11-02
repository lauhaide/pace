import argparse
import numpy as np
from tqdm import tqdm
import argparse
from vllm import LLM, SamplingParams
import sys
sys.path.append('utils/')
from utils import load_file, save_file_jsonl, getChatMessages
import jsonlines
import json
import os

PROMPT_DICT = {
    "prompt_accuracy_eval": (
"You need to check whether the prediction of a question-answering system to a question is correct. You should make the judgment based on a list of "
"ground truth answers provided to you. Your response should be \"correct\" if the prediction matches the ground truth answers and \"incorrect\" otherwise.\n"
"\nQuestion: Who authored The Taming of the Shrew (published in 2002)?"
"\nGround truth: [\"William Shakespeare\", \"Roma Gill\"]"
"\nPrediction: W Shakespeare"
"\nCorrectness: correct\n"
"\nQuestion: Who authored The Taming of the Shrew (published in 2002)?"
"\nGround truth: [\"William Shakespeare\", \"Roma Gill\"]"
"\nPrediction: Roma Gill and W Shakespeare"
"\nCorrectness: correct\n"
"\nQuestion: Who authored The Taming of the Shrew (published in 2002)?"
"\nGround truth: [\"William Shakespeare\", \"Roma Gill\"]"
"\nPrediction: Roma Shakespeare"
"\nCorrectness: incorrect\n"
"\nQuestion: What country is Maharashtra Metro Rail Corporation Limited located in?"
"\nGround truth: [\"India\"]"
"\nPrediction: Maharashtra"
"\nCorrectness: incorrect\n"
"\nQuestion: What's the job of Song Kang-ho in Parasite (2019)?"
"\nGround truth: [\"actor\"]"
"\nPrediction: He plays the role of Kim Ki-taek, the patriarch of the Kim family."
"\nCorrectness: correct\n"
"\nQuestion: Which era did Michael Oakeshott belong to?"
"\nGround truth: [\"20th-century philosophy\"]"
"\nPrediction: 20th century."
"\nCorrectness: correct\n"
"\nQuestion: Edward Tise (known for Full Metal Jacket (1987)) is in what department?"
"\nGround truth: [\"sound department\"]"
"\nPrediction: 2nd Infantry Division, United States Army"
"\nCorrectness: incorrect\n"
"\nQuestion: What wine region is Finger Lakes AVA a part of?"
"\nGround truth: [\"New York wine\"]"
"\nPrediction: Finger Lakes AVA"
"\nCorrectness: incorrect\n"
"\nQuestion: {instruction}"
"\nGround truth: {answers}"
"\nPrediction: {output}"
"\nCorrectness:"
    )
}

def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst


def load_file(input_fp):
    if input_fp.endswith(".json"):
        input_data = json.load(open(input_fp))
    else:
        input_data = load_jsonlines(input_fp)
    return input_data


def save_file_jsonl(data, fp):
    with jsonlines.open(fp, mode='w') as writer:
        writer.write_all(data)

def save_file_json(data, fp):
    with open(fp, 'w') as f:
        json.dump(data, f) 


def getChatMessages(model_name, prompt_name, item):
    if 'Meta-Llama-3.1' in model_name or 'Qwen3' in model_name or 'Qwen2' in model_name:
        return  [
                    { "role" : "system",
                      "content" : "You are a helpful, respectful and honest assistant. "
                    },
                    { "role" : "user",
                      "content": PROMPT_DICT[prompt_name].format_map(item)
                    }
                ]
    elif 'gemma-3' in model_name:
        return  [
                    { "role" : "user",
                      "content": [ {"type": "text", "text": PROMPT_DICT[prompt_name].format_map(item)} ]
                    }

                ]            

def call_model(prompts, model, warmup_caching=None):
    sampling_params = SamplingParams(
        temperature=0.0, top_p=1, max_tokens=5, logprobs=1)
    if warmup_caching==0:
        print('warming up...')
        out = model.generate(prompts[0], sampling_params, use_tqdm=False)
        print(out[0].outputs[0].text, 'warm-up done.')
    preds = model.generate(prompts, sampling_params, use_tqdm=False)

    preds = [pred.outputs[0].text.split("\n\n")[0] for pred in preds]
    postprocessed_preds = [postprocess_output(pred) for pred in preds]

    return postprocessed_preds


def postprocess_output(pred):
    pred = pred.replace("</s>", "")

    if len(pred) > 0 and pred[0] == " ":
        pred = pred[1:]
    return pred

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        default="")
    parser.add_argument('--device', type=str, default="cuda")                  
    parser.add_argument('--home', type=str)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument("--dtype",  type=str, default=None,
                        help="world size to use multiple GPUs.")
    parser.add_argument("--world_size",  type=int, default=1,
                        help="world size to use multiple GPUs.")
    parser.add_argument('--download_dir', type=str, help="specify download dir",
                        default=".cache")
    parser.add_argument('--prompt_name', type=str, default="prompt_accuracy_eval", 
                        help="other value is: 'chat_accuracy_eval-rlhf-calib' used with chat_template")
    parser.add_argument('--chat_template', action="store_true")
    parser.add_argument('--recompute', action="store_true", help="whether it is run for the first time or not.")
    
    args = parser.parse_args()

    # TODO: dictionary w/ outputs to eval could be loaded
    directories = {
    #PACE-GRPO model is saved w/ the following name "qwen3_msq_potgrpo_trajel3_ret02sd005locov01_adapMO005p0609_gd02dec_rs005_pr08"
    "PACE-GRPO": [
        "musique_2026_05_25_09_50_lenrag", "bamboogle_2026_05_25_10_21_lenrag", "hotpotqa_2026_05_25_10_25_lenrag", "2wikimultihopqa_2026_05_25_10_38_lenrag",  \
        "crag607_2026_08_13_16_35_lenrag", "browsecompplus_2026_07_03_17_50_lenrag"
        ],
    }

    print(f'{len(directories)} to evaluate \n {directories.keys()}')

    if args.dtype is not None:
        model = LLM(model=args.model_name, dtype=args.dtype,
                    tensor_parallel_size=args.world_size)
    elif 'Qwen2-72B' in args.model_name and 'AWQ' in args.model_name:
        # https://qwen.readthedocs.io/en/latest/benchmark/speed_benchmark.html
        model = LLM(model=args.model_name, quantization='awq', enable_prefix_caching=False,
                    tensor_parallel_size=args.world_size, trust_remote_code=True,
                    gpu_memory_utilization=0.9, max_model_len=16000, enforce_eager=False)
    else:
        model = LLM(model=args.model_name, 
                    tensor_parallel_size=args.world_size,
                    gpu_memory_utilization=0.9, max_model_len=16000)
    
    summary_directories = {}
    tokenizer = model.get_tokenizer()
    for m in directories.keys():
        summary_directories[m] = {}
    for m in directories.keys():
        print(f'Evaluating: {m}')
        for directory in directories[m]:
            print(f'Dataset: {directory}')

            input_file = args.home + directory + "/intermediate_data.json"
            result_fp =  args.home + directory + "/metric_score.txt"
            if args.recompute:
                input_data = load_jsonlines(input_file)
            else:
                input_data = load_file(input_file)
            print('File uploaded, ', input_file, len(input_data))
            print('Formatting input data...')

            print('Start annotation...')
            for idx in tqdm(range(len(input_data) // args.batch_size)):
                batch = input_data[idx*args.batch_size:(idx+1)*args.batch_size]
                processed_batch = []

                for item in batch:
                    eval_item = {'instruction': item['question'], 'answers': ", ".join(item['golden_answers']), 'output': item['output']['pred']}
                    if args.chat_template:
                        tokenized_chat = tokenizer.apply_chat_template(
                                            getChatMessages(args.model_name, args.prompt_name, eval_item), 
                                            tokenize=False, add_generation_prompt=True)
                        processed_batch.append(tokenized_chat)
                    else:
                        processed_batch.append(PROMPT_DICT[args.prompt_name].format_map(eval_item))

                preds = call_model(processed_batch, model=model)

                preds = [0 if ('incorrect' in pred) or not ('correct' in pred) else 1 for pred in preds]
                
                for j, item in enumerate(batch):
                    item["acc_LM"] = preds[j]
                    if preds[j]==1:
                        print(f"A:{item['golden_answers']} -- P:{item['output']['pred']}")


            if len(input_data) % args.batch_size > 0:
                batch = input_data[(idx+1)*args.batch_size:]
                processed_batch = []

                for item in batch:
                    eval_item = {'instruction': item['question'], 'answers': ", ".join(item['golden_answers']), 'output': item['output']['pred']}
                    if args.chat_template:
                        tokenized_chat = tokenizer.apply_chat_template(
                                            getChatMessages(args.model_name, args.prompt_name, eval_item), 
                                            tokenize=False, add_generation_prompt=True)
                        processed_batch.append(tokenized_chat)
                    else:
                        processed_batch.append(PROMPT_DICT[args.prompt_name].format_map(eval_item))

                preds = call_model(processed_batch, model=model)

                preds = [0 if ('incorrect' in pred) or not ('correct' in pred) else 1 for pred in preds]
                
                for j, item in enumerate(batch):
                    if item['output']['pred'] == "":
                        item["acc_LM"] = 0
                    else:    
                        item["acc_LM"] = preds[j]

            print('Finalised QA, start evaluation...')

            avg_accLM = np.mean([item["acc_LM"] for item in input_data])
            print(f"overall result acc_LM-{args.model_name}: {avg_accLM}")
            with open(result_fp, "a") as f:
                f.write(f"accLM-{args.model_name}: {avg_accLM}\n")
            print('Results saved to, ', result_fp)
            summary_directories[m][directory] = avg_accLM

            save_file_jsonl(input_data, input_file)
            print('Rewrite file with scores to, ', input_file)

    print('Summary accLM results with judge model:', args.model_name)
    for m in summary_directories.keys():
        print(f"{m}")
        for directory in summary_directories[m]:
            print(f"{directory}: {summary_directories[m][directory]}")


if __name__ == "__main__":
    main()