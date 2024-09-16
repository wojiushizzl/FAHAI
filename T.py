import os
import time
import shutil
import ruamel.yaml
import subprocess
import threading
import fnmatch
import uuid

yaml = ruamel.yaml.YAML()



def get_epoch_value(yaml_file):
    # 打开并读取 YAML 文件
    with open(yaml_file, 'r',encoding="utf-8") as file:
        try:
            config = yaml.load(file)  # 解析 YAML 文件内容
            # 获取 epoch 的值
            if 'epochs' in config:
                return config['epochs']
            else:
                raise ValueError("YAML 文件中没有找到 'epoch' 键")
        except Exception as e:
            ValueError(f"读取 YAML 文件时出错: {e}")



epochs = get_epoch_value('./projects/ZZL_Detect/train/train/args.yaml')
print(epochs)