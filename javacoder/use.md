# AutoCodeBench Javacoder

## Configuration Guide

Edit the `src/config.yaml` file to configure the model paths:

```yaml
# CodeLlama model path
codellama7b_repo: "/path/to/CodeLlama-7b"
codellama_max_token: 8192

# DeepSeek model path
deepseekcoder_repo: "/path/to/deepseek-coder-6.7b"
deepseek_max_token: 8192

# Other model configurations...
```

* * *

## Usage Instructions

### 1. Generate the Project Dependency Graph

```bash
cd src
python preprocess_java_ts.py
```

* * *

### 2. Generate Completion Prompts

Edit the `src/utils.py` file to configure training parameters:

```bash
cd src
python main_java.py --timeout 30 --batch_size 100
```

* * *

### 3. Evaluate Model Performance

```bash
cd src
python eval_java_vllm.py --batch_size 8
```

* * *

## Supported Models

* CodeLlama (7B)
    
* DeepSeek-Coder
    
* OpenAI GPT Series (GPT-3.5, GPT-4)
    
* More models can be found in `src/config.yaml`
    

* * *

## Evaluation Metrics

* **Exact Match**: The proportion of generated results that exactly match the ground truth answers
    
* **Edit Similarity**: Similarity computed based on Levenshtein distance
    

* * *

## Notes

* Model paths in the configuration file must be adjusted according to your actual environment
    
* Parsing Java code requires support from the **tree-sitter** library
    
* Large models require sufficient GPU memory
    
* The first run requires preprocessing the project dependency graph, which may take a long time