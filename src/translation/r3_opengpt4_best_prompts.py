import os
import logging
import time
from dotenv import load_dotenv
import argparse
from tqdm import tqdm
from llama_cpp import Llama

os.makedirs(f'logs', exist_ok=True)
logging.basicConfig(filename=f"logs/translation.log", level=logging.INFO, format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


def get_prompt_response(llm, prompt_type, input_code, source, to):
    if prompt_type == 'original-baseline':
        content = "".join(input_code) + f"\n# Translate the above {source} code to {to}. Print only the {to} code and end with the comment \"End of Code\".\n"

        message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content}]

        return llm.create_chat_completion(message)['choices'][0]['message']['content']

    elif prompt_type == 'revamped-cotnl':
        content = "".join(input_code) + f"\n# Translate the above {source} code to {to}. Print only the {to} code and end with the comment \"End of Code\". Let's think step by step.\n"

        sample_input = "Example: N, K = map(int, input().split())\nif N % K == 0:\n\tprint(0)\nelse:\n\tprint(1)\n\" Translate the above Python code to C."

        sample_output = "Response: // The program reads two integers N and K from the user input and then checks whether N is divisible by K. It outputs 0 if true, otherwise outputs 1.\n main(){\nint N, K;\nscanf(\"%d %d\", &N, &K);\nif (N % K == 0){ \n\tprintf(\"0\n\");\n }\nelse{\n\tprintf(\"1\n\");\n}\nreturn 0;}\n// End of Code"

        message = [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content + "\n" + sample_input + "\n" + sample_output}]

        return llm.create_chat_completion(message)['choices'][0]['message']['content']

    elif prompt_type == 'revamped-cotnl-ast':
        content = "".join(input_code) + f"\n# Translate the above {source} code to {to}. Print only the {to} code and end with the comment \"End of Code\". Let's think step by step.\n"

        sample_input = "Example: N, K = map(int, input().split())\nif N % K == 0:\n\tprint(0)\nelse:\n\tprint(1)\n\" Translate the above Python code to C."

        sample_output = "Response: // The program reads two integers N and K from the user input and then checks whether N is divisible by K. It outputs 0 if true, otherwise outputs 1.\n // Assignment -> Tuple (N, K) -> Function Call (map) -> Function (int), Function Call (input) -> Method Call (split)\n If Statement -> Condition (Equality Check) -> Modulo Operation (N % K), Value 0 -> If Branch -> Function Call (print) with argument 0 -> Else Branch -> Function Call (print) with argument 1\n main(){\nint N, K;\nscanf(\"%d %d\", &N, &K);\nif (N % K == 0){ \n\tprintf(\"0\n\");\n }\nelse{\n\tprintf(\"1\n\");\n}\nreturn 0;}\n// End of Code"

        message = [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content + "\n" + sample_input + "\n" + sample_output}]

        return llm.create_chat_completion(message)['choices'][0]['message']['content']

    return ""


def main(args):

    extensions = { 'Python': 'py','C': 'c','C++': 'c++','Java': 'java','Go': 'go', "Rust": "rs", "C#": "cs" }

    in_folder = f'dataset/{args.dataset}/{args.source_lang}/Code'
    out_folder = f'output/OpenGPT-4/{args.dataset}/{args.source_lang}/{args.target_lang}'

    in_files = os.listdir(in_folder)
    print(f'found {len(in_files)} inputs')

    # check for files alraedy extracted
    already_extracted_files = []
    if os.path.exists(out_folder):
        already_extracted_files = os.listdir(out_folder)
        if len(already_extracted_files) > 0:
            already_extracted_files = [x.split('.')[0] for x in already_extracted_files if os.stat(f'{out_folder}/{x}').st_size != 0]

    if len(already_extracted_files) > 0:
        in_files = [x for x in in_files if x.split('.')[0] not in already_extracted_files]

    ext = extensions[args.target_lang]
        
    llm = Llama(
        model_path="./open_gpt4_8x7b.Q4_K_M.gguf", 
        chat_format="llama-2", 
        cache_dir='./huggingface',
        n_gpu_layers=35,
        n_ctx=32768,
    )  # Set chat_format according to the model you are using

    # loop over input files
    os.makedirs(out_folder, exist_ok=True)
    for f in tqdm(in_files):
        prompt_file = f'{in_folder}/{f}'

        with open(prompt_file, "r", encoding="ISO-8859-1", errors='ignore') as fin:
            input_code = fin.readlines()

            try:
                t0 = time.perf_counter()
                source = args.source_lang
                to = args.target_lang

                response = get_prompt_response(llm, args.prompt_type, input_code, source, to)
                
                print(response)
                response = response.replace(f"```{to.lower()}", "").replace("```", "")

                t1 = time.perf_counter()
                print("Total generation time:", t1 - t0)
                out_file = f'{out_folder}/{f.split(".")[0]}.{ext}'                
                with open(out_file, 'w') as fot:
                    print(response, file=fot)

            except (ValueError, FileNotFoundError) as e:
                print(e)
                continue


if __name__ == "__main__":

    load_dotenv()

    parser = argparse.ArgumentParser(description='run translation with open GPT-4 with a given dataset and languages')
    parser.add_argument('--dataset', help='dataset to use for code translation. should be one of [codenet,avatar]', required=True, type=str)
    parser.add_argument('--source_lang', help='source language to use for code translation. should be one of [Python,Java,C,C++,Go]', required=True, type=str)
    parser.add_argument('--target_lang', help='target language to use for code translation. should be one of [Python,Java,C,C++,Go]', required=True, type=str)
    parser.add_argument('--k', help='The number of highest probability vocabulary tokens to keep for top-k-filtering. Only applies for sampling mode, with range from 1 to 100.', required=True, type=int)
    parser.add_argument('--p', help='Only the most probable tokens with probabilities that add up to top_p or higher are considered during decoding. The valid range is 0.0 to 1.0. 1.0 is equivalent to disabled and is the default. Only applies to sampling mode. Also known as nucleus sampling.', required=True, type=float)
    parser.add_argument('--temperature', help='A value used to warp next-token probabilities in sampling mode. Values less than 1.0 sharpen the probability distribution, resulting in "less random" output. Values greater than 1.0 flatten the probability distribution, resulting in "more random" output. A value of 1.0 has no effect and is the default. The allowed range is 0.0 to 2.0.', required=True, type=float)
    parser.add_argument('--prompt_type', help='which prompt type to use. should be one of [original-baseline, revamped-cotnl, revamped-cotnl-ast]', required=True, type=str)
    args = parser.parse_args()

    source = args.source_lang
    target = args.target_lang

    logging.info(f"translating examples from {source} to {target} using OpenGPT-4 and {args.dataset} dataset")
    main(args)
