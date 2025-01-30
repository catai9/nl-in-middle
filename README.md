# nl-in-middle

This repository adapts code from *Lost in Translation: A Study of Bugs Introduced by Large Language Models while Translating Code* for the studied open-source models and datasets.

### Install
Create a virtual environment by running the following commands. We downloaded `conda 23.11.0` as per recommendation from the other repository.

```
conda create -n nl-in-middle python=3.10.13
conda activate nl-in-middle
```

Install the software dependencies by running

```
pip3 install -r requirements.txt
```

For compiling and testing the generated translations, we used the same configuration as Lost in Translation (specifically: Python 3.10, g++ 11, GCC Clang 14.0, Java 11, Go 1.20, and .Net 7.0.14 for Python, C++, C, Java, Go, and C# respectively).

Download the dataset from *Lost in Translation: A Study of Bugs Introduced by Large Language Models while Translating Code* at [Zenodo](https://zenodo.org/doi/10.5281/zenodo.8190051). 


### Running Scripts
**Translation**
Example of how to run the translation scripts for the various models are below.
```
# CodeGen
python3 src/translation/r3_codegen_revamp_prompts.py --dataset codenet --source_lang C --target_lang C++ --k 50 --p 0.95 --temperature 0.2 --prompt_type revamped-cot-nl --gpu_id 0;

# OpenGPT-4
python3 src/translation/r3_codegen_revamp_prompts.py --dataset codenet --source_lang C --target_lang C++ --k 50 --p 0.95 --temperature 0.2 --prompt_type revamped-cot-nl --gpu_id 0;

# StarCoder
python3 src/translation/r3_starcoder_revamp_prompts.py --dataset codenet --source_lang C --target_lang C++ --k 50 --p 0.95 --temperature 0.2 --prompt_type revamped-cot-nl --gpu_id 0;
```

**Test**
Example of how to run the test scripts for the various datasets are below.
```
# Clean generations
python3 src/translation/temp_clean_generations.py --model StarCoder --dataset CodeNet --prompt cot-nl;

# CodeNet 
python3 src/test/compile_codenet.py --source_lang Python --target_lang Java --model StarCoder --report_dir report;
python3 src/test/compile_codenet_feedback.py --source_lang Python --target_lang Java --model StarCoder --report_dir report --attempt 1;

# AVATAR
python3 src/test/compile_avatar.py --source_lang Python --target_lang Java --model StarCoder --report_dir report;
python3 src/test/compile_avatar_feedback.py --source_lang Python --target_lang Java --model StarCoder --report_dir report --attempt 1;
```
