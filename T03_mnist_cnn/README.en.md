语言：[简体中文 🇨🇳](README.md) | Language: English 🇺🇸

# PyTorch MNIST Handwritten Digit Recognition (3)

This folder contains the code used in the article [Learn PyTorch by Examples (3): MNIST Handwritten Digit Recognition (III) -- Convolutional Neural Networks](https://blog.jinli.io/en/p/learn-pytorch-by-examples-3-mnist-handwritten-digit-recognition-iii--convolutional-neural-networks/).

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

    ```bash
    python plot_performance.py
    ```

3. **Recognize handwritten digits**:

    You can create a handwritten digit image to recognize, and then run `classify.py` to recognize this image. I provide some example images that I wrote in the `numbers` folder, you can use these images to test.

    ```bash
    python classify.py numbers/number0.png
    ```