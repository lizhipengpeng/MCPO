# MCPO
## Overview

Code for paper "User Behavior Alignment via Large Language Model with Model Cooperation and Parameter Optimization"

## Code Structure

├── data_generate.ipynb // convert data format
├── LICENSE
├── local_evaluation.py  // main function
├── metrics.py  // metrics function
├── models
│   ├── base_model.py
│   ├── dummy_model.py
│   ├── little_model.py
│   ├── model_best.py
│   ├── user_config.py
│   └── vanilla_llama3_baseline.py // LLM function
├── parsers.py
├── README.md
├── requirements_eval.txt  //requirements for metrics
├── requirements.txt // requirements for LLM
├── run_result.sh // run script
└── utilities
    └── _Dockerfile

## Dependencies

python and other requirements in requirements.txt

## Running the Code

python local_evaluation.py --model_path /model/path --/data/path  --instruction {self_cot_mcp|self_cot_all|little_model_cot_mcp|little_model_cot_all|generation_prompt|similarity|all}

## Contact

