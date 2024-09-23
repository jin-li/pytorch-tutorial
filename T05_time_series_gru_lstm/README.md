语言：简体中文 🇨🇳 | Language: [English 🇺🇸](README.en.md)

# PyTorch 序列预测（二）

本文件夹包含了[《实例学PyTorch（5）：序列预测（二）——门控循环单元（GRU）和长短期记忆网络（LSTM）》](https://jinli.io/p/%E5%AE%9E%E4%BE%8B%E5%AD%A6pytorch5%E5%BA%8F%E5%88%97%E9%A2%84%E6%B5%8B%E4%BA%8C%E9%97%A8%E6%8E%A7%E5%BE%AA%E7%8E%AF%E5%8D%95%E5%85%83gru%E5%92%8C%E9%95%BF%E7%9F%AD%E6%9C%9F%E8%AE%B0%E5%BF%86%E7%BD%91%E7%BB%9Clstm/)文章中使用的代码。

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

        或者

        ```bash
        python time_series_models.py --generate-data
        ```
    
    - 运行模型，绘制性能曲线和预测值：
    
        ```bash
        python compare_results.py
        ```