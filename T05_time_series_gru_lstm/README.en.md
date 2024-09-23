语言：[简体中文 🇨🇳](README.md) | Language: English 🇺🇸

# PyTorch Series Prediction (2)

This folder contains the code used in the article [Learn PyTorch by Examples (5): Sequence Prediction (II) -- Gated Recurrent Unit (GRU) and Long Short-Term Memory (LSTM)](https://jinli.io/en/p/learn-pytorch-by-examples-5-sequence-prediction-ii--gated-recurrent-unit-gru-and-long-short-term-memory-lstm/).

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

        or

        ```bash
        python time_series_models.py --generate-data
        ```

    - Run the model, plot performance curves and predicted values:
    
        ```bash
        python compare_results.py
        ```