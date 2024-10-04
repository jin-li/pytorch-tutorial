语言：[简体中文 🇨🇳](README.md) | Language: English 🇺🇸

# PyTorch Language Model (II)

This folder contains the code used in the article [Learn PyTorch by Examples (7): Language Model (II) - Implement Word-Level Language Model with Transformer](https://jinli.io/en/p/learn-pytorch-by-examples-7-language-model-ii-implement-word-level-language-model-with-transformer/).

## Quick Start

First, you need to create a Python virtual environment to run this project. You can use `virtualenv` or `conda` and other tools to create a virtual environment. You can refer to my article [Python Environment Management with venv/conda/mamba](https://jinli.io/en/p/python-environment-management-with-venv/conda/mamba/).

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the code:

    - Download data:
    
        ```bash
        python download_WikiText2.py
        ```

    - Run the model, plot performance curves:
    
        ```bash
        python language_transformer.py --plot
        ```

        - By default, it trains 50 epochs using CUDA. If CUDA is not available, it uses MPS or CPU for training. Training may take a long time, you can interrupt the training with Ctrl+C.
        - You can also specify training parameters through command line arguments, see the code or use `python language_transformer.py -h` for help.
    
    - Generate text:
    
        ```bash
        python generate_text.py
        ```
