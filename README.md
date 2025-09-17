# NL in the Middle: Code Translation with LLMs and Intermediate Representations
This repository adapts code from *Lost in Translation: A Study of Bugs Introduced by Large Language Models while Translating Code* [url](https://github.com/Intelligent-CAT-Lab/PLTranslationEmpirical) for the studied open-source models and datasets. Subsequently, we provide similar setup and data download instructions from that repository as well.

[![Preprint](https://img.shields.io/badge/read-preprint-blue)](https://arxiv.org/abs/2507.08627)
[![Install](https://img.shields.io/badge/install-instructions-blue)](README.md#install)
[![Dependencies](https://img.shields.io/badge/install-dependencies-blue)](README.md#dependencies)
[![Scripts](https://img.shields.io/badge/run-scripts-blue)](README.md#scripts)
[![Data](https://zenodo.org/badge/DOI/10.5281/zenodo.8190051.svg)](https://zenodo.org/doi/10.5281/zenodo.8190051)

Artifact repository for the paper [_NL in the Middle: Code Translation with LLMs and Intermediate Representations_](https://arxiv.org/abs/2507.08627), accepted at _CASCON 2025_, Toronto, Canada.
Authors are Chi-en Amy Tai, Pengyu Nie, Lukasz Golab, and Alexander Wong.


### Install
In this repository, we provide the source code required to replicate the findings from our paper. Start by cloning the repository here:
```
git clone https://github.com/catai9/nl-in-middle
```

To run the scripts, we used a virtual environment. We recommend downloading `conda 23.11.0` from this [link](https://docs.conda.io/projects/miniconda/en/latest/miniconda-other-installer-links.html) and creating a virtual environment using the following command:
```
conda create -n nl-in-middle python=3.10.13
conda activate nl-in-middle # This activates the virtual environment 
```

### Dependencies
To install all software dependencies, please execute the following commands:
```
python3 --version && pip3 --version # Check Python version correct (should be 3.10.13)
pip3 install -r requirements.txt
```

For running experiments, we used three NVIDIA RTX 6000 GPUs, with 51.5 GB memory each for inferencing the models. Given the smaller sizes of our models, a single NVIDIA RTX 6000 GPU (or better GPU version) could also be used as long as it has sufficient GB memory to store and run inference on the models. 

For compiling and testing the generated translations, we used the same configuration as Lost in Translation (specifically: Python 3.10, g++ 11, GCC Clang 14.0, Java 11, Go 1.20, and .Net 7.0.14 for Python, C++, C, Java, Go, and C# respectively).

### Dataset
We used the same CodeNet and AVATAR datasets from *Lost in Translation: A Study of Bugs Introduced by Large Language Models while Translating Code* at [Zenodo](https://zenodo.org/doi/10.5281/zenodo.8190051). 

After downloading the `dataset.zip` file from Zenodo, unzip the file and confirm that you see the codenet and avatar datasets:

```
PLTranslationEmpirical
├── dataset
    ├── codenet
    ├── avatar
├── ...
```

In these two folders, there are folders that correspond to a source language and within each folder, there are two directories: `Code` and `TestCases` which respectively contain the code snippets (with an `id` in the filename) and test cases. 

### Scripts
Python scripts are provided to reproduce the results in our paper. Before running the scripts, first create a `.env` file in the repository and add the following: 

```
STARCODER_AUTH_TOKEN=<your starcoder auth token from huggingface>
```

1. Translation: An example of how to run the translation scripts for the various models are included below. The following commands translate all `C -> C++` code snippets in the `codenet` dataset with the three evaluated LLMs using GPU 0. Please adjust the arguments accordingly to obtain results for other code translation pairs or dataset.
```
# CodeGen
python3 src/translation/r3_codegen_revamp_prompts.py --dataset codenet --source_lang C --target_lang C++ --k 50 --p 0.95 --temperature 0.2 --prompt_type revamped-cot-nl --gpu_id 0;

# OpenGPT-4
python3 src/translation/r3_codegen_revamp_prompts.py --dataset codenet --source_lang C --target_lang C++ --k 50 --p 0.95 --temperature 0.2 --prompt_type revamped-cot-nl --gpu_id 0;

# StarCoder
python3 src/translation/r3_starcoder_revamp_prompts.py --dataset codenet --source_lang C --target_lang C++ --k 50 --p 0.95 --temperature 0.2 --prompt_type revamped-cot-nl --gpu_id 0;
```

2. Testing: An example of how to run the testing scripts for the various models are included below. First, the translated codes need to be cleaned (by removing any systematic excess noise). Then the `compile` and `compile_feedback` scripts need to be run for the respective datasets. The sample scripts are for `Python -> Java` code snippets for `StarCoder` LLM. Please adjust the arguments accordingly to obtain test metrics for other code translation pairs. 
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

### Contact
We look forward to hearing your feedback. Please contact [Amy Tai](mailto:amy.tai@uwaterloo.ca) for any questions or comments.

