语言：[简体中文 🇨🇳](README.md) | Language: English 🇺🇸

# PyTorch Language Model (I)

This folder contains the code used in the article [Learn PyTorch by Examples (6): Language Model (I) -- Implementing a Word-Level Language Model with LSTM](https://blog.jinli.io/en/p/learn-pytorch-by-examples-6-language-model-i--implementing-a-word-level-language-model-with-lstm/).

## Quick Start

First, you need to create a Python virtual environment to run this project. You can use `virtualenv` or `conda` and other tools to create a virtual environment. You can refer to my article [Python Environment Management with venv/conda/mamba](https://blog.jinli.io/en/p/python-environment-management-with-venv/conda/mamba/).

1. **Install dependencies**, here are two ways:

   - `conda`/`mamba`/`micromamba`/`miniconda` (recommended):

     ```bash
     conda env create -f environment.yml
     ```

     If you use `mamba` or `micromamba`, please replace `conda` in the above command with `mamba` or `micromamba`.

   - `pip`:

     If you use `pip`, please make sure the Python version is 3.11. I have tried Python 3.12, and it will report a version mismatch error when installing `torchvision`. I have not tested lower versions of Python.

     ```bash
     pip install -r requirements.txt
     ```

2. **Run the code**:

    - Download data:
    
        ```bash
        python download_WikiText2.py
        ```

    - Run the model, plot performance curves:
    
        ```bash
        python language_lstm.py --plot
        ```

        - By default, it trains 50 epochs using CUDA. If CUDA is not available, it uses MPS or CPU for training. Training may take a long time, you can interrupt the training with Ctrl+C.
        - You can also specify training parameters through command line arguments, see the code or use `python language_lstm.py -h` for help.
    
    - Generate text:
    
        ```bash
        python generate_text.py
        ```

        You can specify parameters for generating text through command line arguments, see the code or use `python generate_text.py -h` for help.
