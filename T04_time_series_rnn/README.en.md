语言：[简体中文 🇨🇳](README.md) | Language: English 🇺🇸

# PyTorch Series Prediction (1)

This folder contains the code used in the article [Learn PyTorch by Example (4): Sequence Prediction with Recurrent Neural Networks (I)](https://jinli.io/en/p/learn-pytorch-by-example-4-sequence-prediction-with-recurrent-neural-networks-i/).

## Quick Start

First, you need to create a Python virtual environment to run this project. You can use `virtualenv` or `conda` and other tools to create a virtual environment. You can refer to my article [Python Environment Management with venv/conda/mamba](https://jinli.io/en/p/python-environment-management-with-venv/conda/mamba/).

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the code:

    - Generate data:

        ```bash
        python SineWaveDataset.py
        ```

    - Run the model:
    
        ```bash
        python time_series_rnn.py --plot
        ```