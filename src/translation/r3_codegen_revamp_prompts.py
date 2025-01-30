import os
import logging
from dotenv import load_dotenv
import time
import argparse
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


os.makedirs(f'logs', exist_ok=True)
logging.basicConfig(filename=f"logs/translation.log", level=logging.INFO, format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def get_prompt_response(model, tokenizer, prompt_type, input_code, source, to, device, out_folder, ext, p, k, temperature, f):
    if prompt_type == 'original-baseline':
        prompt = f"{args.source_lang} Code:\n\n" + "".join(input_code) + f'\n\nTranslate the above {args.source_lang} code to {args.target_lang}.\n\n{args.target_lang} Code:\n\n'

        inputs = []
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

        total_input_tokens = inputs.shape[1]
        model_max_length = 2048
        if total_input_tokens >= model_max_length:
            out_file = f'{out_folder}/{f.split(".")[0]}.{ext}'
            with open(out_file, 'w') as fot:
                print("# Token size exceeded.", file=fot)
            return
        max_new_tokens = model_max_length - total_input_tokens

        raw_outputs = ''
        
        raw_outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=p,
            top_k=k,
            temperature=temperature,
            repetition_penalty=1,
            pad_token_id=tokenizer.eos_token_id,
        )

        return tokenizer.decode(raw_outputs[0])

    elif prompt_type == 'revamped-cotnl':
        example_python_code = 'N, K = map(int, input().split())\nif N % K == 0:\n\tprint(0)\nelse:\n\tprint(1)\n'

        example_c_code = '// Summary: The program reads two integers N and K from the user input and then checks whether N is divisible by K. It outputs 0 if true, otherwise outputs 1.\n\nmain(){\nint N, K;\nscanf(\"%d %d\", &N, &K);\nif (N % K == 0){ \n\tprintf(\"0\n\");\n }\nelse{\n\tprintf(\"1\n\");\n}\nreturn 0;}\n'

        prompt = f"Python Code:\n\n" + "".join(example_python_code) + f'\n\nTranslate the above Python code to C. Let\'s think step by step.\n\nC Code:\n\n' + "".join(example_c_code) + "\n\n" + f"{source} Code:\n\n" + "".join(input_code) + f'\n\nTranslate the above {source} code to {to}.\n\n{to} Code:\n\n'

        inputs = []
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

        total_input_tokens = inputs.shape[1]
        model_max_length = 2048
        if total_input_tokens >= model_max_length:
            out_file = f'{out_folder}/{f.split(".")[0]}.{ext}'
            with open(out_file, 'w') as fot:
                print("# Token size exceeded.", file=fot)
            return
        max_new_tokens = model_max_length - total_input_tokens

        raw_outputs = ''
        
        raw_outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=p,
            top_k=k,
            temperature=temperature,
            repetition_penalty=1,
            pad_token_id=tokenizer.eos_token_id,
        )

        return tokenizer.decode(raw_outputs[0])
    

    elif prompt_type == 'revamped-cotnl-ast':
        example_python_code = 'N, K = map(int, input().split())\nif N % K == 0:\n\tprint(0)\nelse:\n\tprint(1)\n'

        example_nl = '// Summary: The program reads two integers N and K from the user input and then checks whether N is divisible by K. It outputs 0 if true, otherwise outputs 1.\n\n'

        example_ast = '// AST: Assignment -> Tuple (N, K) -> Function Call (map) -> Function (int), Function Call (input) -> Method Call (split)\n If Statement -> Condition (Equality Check) -> Modulo Operation (N % K), Value 0 -> If Branch -> Function Call (print) with argument 0 -> Else Branch -> Function Call (print) with argument 1\n\n'

        example_c_code = 'main(){\nint N, K;\nscanf(\"%d %d\", &N, &K);\nif (N % K == 0){ \n\tprintf(\"0\n\");\n }\nelse{\n\tprintf(\"1\n\");\n}\nreturn 0;}\n'

        prompt = f"Python Code:\n\n" + "".join(example_python_code) + f'\n\nTranslate the above Python code to C. Let\'s think step by step.\n\nC Code:\n\n' + "".join(example_nl) + "".join(example_ast) + "".join(example_c_code) + "\n\n" + f"{source} Code:\n\n" + "".join(input_code) + f'\n\nTranslate the above {source} code to {to}.\n\n{to} Code:\n\n'

        inputs = []
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

        total_input_tokens = inputs.shape[1]
        model_max_length = 2048
        if total_input_tokens >= model_max_length:
            out_file = f'{out_folder}/{f.split(".")[0]}.{ext}'
            with open(out_file, 'w') as fot:
                print("# Token size exceeded.", file=fot)
            return
        max_new_tokens = model_max_length - total_input_tokens

        raw_outputs = ''
        
        raw_outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=p,
            top_k=k,
            temperature=temperature,
            repetition_penalty=1,
            pad_token_id=tokenizer.eos_token_id,
        )

        return tokenizer.decode(raw_outputs[0])

    return ""


def main(args):

    extensions = { 'Python': 'py','C': 'c','C++': 'c++','Java': 'java','Go': 'go', "Rust": "rs", "C#": "cs" }

    in_folder = f'dataset/{args.dataset}/{args.source_lang}/Code'
    out_folder = f'output/CodeGen/{args.dataset}/{args.source_lang}/{args.target_lang}'

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
    device = f'cuda:{args.gpu_id}'

    tokenizer, model = None, None
    
    kwargs = {}
    kwargs["torch_dtype"] = torch.float16
    tokenizer = AutoTokenizer.from_pretrained('Salesforce/codegen-16B-multi', cache_dir='./huggingface')
    model = AutoModelForCausalLM.from_pretrained('Salesforce/codegen-16B-multi', cache_dir='./huggingface', **kwargs).to(device)
    
    # loop over input files
    os.makedirs(out_folder, exist_ok=True)
    for f in tqdm(in_files):
        prompt_file = f'{in_folder}/{f}'

        with open(prompt_file, "r", encoding="ISO-8859-1", errors='ignore') as fin:
            prompt = fin.readlines()
            
            try:
                t0 = time.perf_counter()

                result = get_prompt_response(model, tokenizer, args.prompt_type, prompt, args.source_lang, args.target_lang, device, out_folder, ext, args.p, args.k, args.temperature, f)
                
                if result == None:
                    continue
                data = result[result.find(f'{args.target_lang} Code:')+len(f'{args.target_lang} Code:'):].strip()        

                valid_lines = []
                for line in data.split('\n'):
                    if line.strip() in ["*/", 'C Code:', 'C++ Code:', 'Java Code:', 'Python Code:', 'Go Code:']:
                        break
                    else:
                        valid_lines.append(line)

                data = '\n'.join(valid_lines)
                data = data.replace('<|endoftext|>', '')

                t1 = time.perf_counter()
                print("Total generation time:", t1 - t0)
                out_file = f'{out_folder}/{f.split(".")[0]}.{ext}'                
                with open(out_file, 'w') as fot:
                    print(data, file=fot)

            except (ValueError, FileNotFoundError) as e:
                print(e)
                continue


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description='run translation with open-source models given dataset and languages')
    parser.add_argument('--dataset', help='dataset to use for code translation. should be one of [codenet,avatar,evalplus]', required=True, type=str)
    parser.add_argument('--source_lang', help='source language to use for code translation. should be one of [Python,Java,C,C++,Go]', required=True, type=str)
    parser.add_argument('--target_lang', help='target language to use for code translation. should be one of [Python,Java,C,C++,Go]', required=True, type=str)
    parser.add_argument('--k', help='The number of highest probability vocabulary tokens to keep for top-k-filtering. Only applies for sampling mode, with range from 1 to 100.', required=True, type=int)
    parser.add_argument('--p', help='Only the most probable tokens with probabilities that add up to top_p or higher are considered during decoding. The valid range is 0.0 to 1.0. 1.0 is equivalent to disabled and is the default. Only applies to sampling mode. Also known as nucleus sampling.', required=True, type=float)
    parser.add_argument('--temperature', help='A value used to warp next-token probabilities in sampling mode. Values less than 1.0 sharpen the probability distribution, resulting in "less random" output. Values greater than 1.0 flatten the probability distribution, resulting in "more random" output. A value of 1.0 has no effect and is the default. The allowed range is 0.0 to 2.0.', required=True, type=float)
    parser.add_argument('--gpu_id', help='gpu id to use', required=True, type=int)
    parser.add_argument('--prompt_type', help='which prompt type to use. should be one of [original-baseline, revamped-cotnl, revamped-cotnl-ast]', required=True, type=str)
    args = parser.parse_args()

    # Initialize configurations
    source = args.source_lang
    target = args.target_lang
    logging.info(f"translating examples from {source} to {target} using CodeGen and {args.dataset} dataset")
    main(args)
