# AutoCodeBench

## Project Overview

This project is an intelligent code completion system based on large language models (LLMs).  
By analyzing project structure, function dependencies, and type definitions, the system generates context-aware prompts for code completion tasks, significantly improving the accuracy and relevance of code generation.

* * *

## Prepare Environment

```bash
conda create -n AutoCodeBench python=3.10
pip install -r requirements.txt
```

* * *

## Dataset

Each sample in the Benchmark datasets (`Benchmark/c_metadata.jsonl` or `Benchmark/java_metadata.jsonl`) contains the following fields:

* `id`: task ID
    
* `pkg`: the repository the sample belongs to
    
* `fpath`: the file path of the code to be completed
    
* `input`: the content before the cursor position
    
* `gt`: the ground truth code line to be completed
    

### Repository Data

The full dataset will be released soon.  
Please download the complete repositories separately and place them into:

* `ccoder/c_repo`
    
* `javacoder/java_repo`
    

* * *

## Usage

For C and Java evaluation, please enter the corresponding module directory:

```bash
cd ccoder
```

```bash
cd javacoder
```

Detailed evaluation instructions can be found in each module’s README.

* * *

## Project Structure

```bash
AutoCodeBench/
├── Benchmark/                 # Benchmark metadata files
│   ├── c_metadata.jsonl       # C completion benchmark dataset
│   └── java_metadata.jsonl    # Java completion benchmark dataset
│
├── ccoder/                    # C code auto-completion system
│   ├── c_repo/                # C source projects used for evaluation
│   ├── README.md              # Documentation for Ccoder module
│   └── src/                   # Core source code for C completion
│
├── javacoder/                 # Java code auto-completion system
│   ├── java_repo/             # Java source projects used for evaluation
│   ├── README.md              # Documentation for Javacoder module
│   └── src/                   # Core source code for Java completion
│
├── README.md                  # Main project documentation
└── requirements.txt           # Python dependencies
```

* * *

## License

This project is intended for academic research purposes.  
For licensing details, please refer to the root directory of the project.