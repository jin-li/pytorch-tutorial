语言：简体中文 🇨🇳 | Language: [English 🇺🇸](README.en.md)

# PyTorch 序列预测（一）

本文件夹包含了[《实例学PyTorch（4）：序列预测（一）——循环神经网络（RNN）》](https://jinli.io/p/%E5%AE%9E%E4%BE%8B%E5%AD%A6pytorch4%E5%BA%8F%E5%88%97%E9%A2%84%E6%B5%8B%E4%B8%80%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9Crnn/)文章中使用的代码。

## 快速开始

首先你需要创建一个运行本项目的Python虚拟环境，你可以使用`virtualenv`或者`conda`等工具来创建虚拟环境。可参考我的文章[《Python环境管理方式总结》](https://jinli.io/p/python%E7%8E%AF%E5%A2%83%E7%AE%A1%E7%90%86%E6%96%B9%E5%BC%8F%E6%80%BB%E7%BB%93/)。

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 运行代码：

    - 生成数据：
    
        ```bash
        python SineWaveDataset.py
        ```
    
    - 运行模型：
    
        ```bash
        python time_series_rnn.py --plot
        ```