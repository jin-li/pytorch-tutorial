语言：简体中文 🇨🇳 | Language: [English 🇺🇸](README.en.md)

# PyTorch 语言模型（二）

本文件夹包含了[实例学PyTorch（7）：语言模型（二）——使用Transformer实现词级语言模型](https://jinli.io/p/%E5%AE%9E%E4%BE%8B%E5%AD%A6pytorch7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E4%BA%8C%E4%BD%BF%E7%94%A8transformer%E5%AE%9E%E7%8E%B0%E8%AF%8D%E7%BA%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/)文章中使用的代码。

## 快速开始

首先你需要创建一个运行本项目的Python虚拟环境，你可以使用`virtualenv`或者`conda`等工具来创建虚拟环境。可参考我的文章[《Python环境管理方式总结》](https://jinli.io/p/python%E7%8E%AF%E5%A2%83%E7%AE%A1%E7%90%86%E6%96%B9%E5%BC%8F%E6%80%BB%E7%BB%93/)。

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 运行代码：

    - 下载数据：
    
        ```bash
        python download_WikiText2.py
        ```

    - 运行模型，绘制性能曲线：
    
        ```bash
        python language_transformer.py --plot
        ```

        - 默认使用CUDA训练50个epoch。如果没有CUDA，则使用MPS或CPU训练。训练时间可能较长，你可以使用Ctrl+C中断训练。
        - 你也可以通过命令行参数指定训练参数，具体参见代码或者使用`python language_transformer.py -h`查看帮助。
    
    - 生成文本：
    
        ```bash
        python generate_text.py
        ```