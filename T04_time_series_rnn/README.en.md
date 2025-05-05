语言：[简体中文 🇨🇳](README.md) | Language: English 🇺🇸

# PyTorch Series Prediction (1)

This folder contains the code used in the article [Learn PyTorch by Examples (4): Sequence Prediction (I) -- Recurrent Neural Networks (RNN)](https://blog.jinli.io/en/p/learn-pytorch-by-examples-4-sequence-prediction-i--recurrent-neural-networks-rnn/).

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

    - Generate data:

        ```bash
        python SineWaveDataset.py
        ```

    - Run the model:
    
        ```bash
        python time_series_rnn.py --plot
        ```