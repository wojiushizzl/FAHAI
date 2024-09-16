from ultralytics import YOLO
from multiprocessing import Process, freeze_support
import os


def train(
        name='test',
        project_name="1_Segment",#保存训练结果的项目目录名称。允许有组织地存储不同的实验。
        epochs=50, #训练历元总数。每个历元代表对整个数据集进行一次完整的训练。调整该值会影响训练时间和模型性能。
        batch=2, #训练的批量大小，表示在更新模型内部参数之前要处理多少张图像。自动批处理 (batch=-1)会根据 GPU 内存可用性动态调整批处理大小。
        patience=100,#在验证指标没有改善的情况下，提前停止训练所需的历元数。当性能趋于平稳时停止训练，有助于防止过度拟合。
        exist_ok=True,#如果为 True，则允许覆盖现有的项目/名称目录。这对迭代实验非常有用，无需手动清除之前的输出。
        single_cls=True,#在训练过程中将多类数据集中的所有类别视为单一类别。适用于二元分类任务，或侧重于对象的存在而非分类。
        imgsz=640, # 用于训练的目标图像尺寸。所有图像在输入模型前都会被调整到这一尺寸。影响模型精度和计算复杂度。

):
    current_directory = os.getcwd()
    project_path = os.path.join(current_directory,'projects', project_name)
    yaml_path = os.path.join(project_path, 'train.yaml')
    result_path = os.path.join(project_path, 'train')
    # Load a model
    current_directory = os.getcwd()
    seg_filepath = os.path.join(current_directory, 'pre_model','yolov8n-seg.pt')
    model = YOLO(seg_filepath)  # load a pretrained model (recommended for training)
    # Train the model
    results = model.train(
        name=name,
        data=yaml_path,
        project=result_path,
        epochs=epochs,
        batch=batch,
        patience=patience,
        exist_ok=exist_ok,
        single_cls=single_cls,
        imgsz=imgsz,
        amp=True
    )


if __name__ == "__main__":
    freeze_support()

    p = Process(target=train)
    p.start()
    p.join()
