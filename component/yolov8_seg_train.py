from ultralytics import YOLO
from multiprocessing import Process, freeze_support
import os
import argparse

def train():
    # 创建解析器
    parser = argparse.ArgumentParser(description="Process some parameters.")

    # 添加参数
    parser.add_argument('name', type=str,  nargs='?',default='test',help='The train name')
    parser.add_argument('project_name', type=str,  nargs='?',default='1_Segment',help='The project_name')
    parser.add_argument('epochs', type=int,  nargs='?',default=50,help='epochs')     # 训练历元总数。每个历元代表对整个数据集进行一次完整的训练。调整该值会影响训练时间和模型性能。
    parser.add_argument('batch', type=int, nargs='?',default=2,help='batch')    # 训练的批量大小，表示在更新模型内部参数之前要处理多少张图像。自动批处理 (batch=-1)会根据 GPU 内存可用性动态调整批处理大小。
    parser.add_argument('patience', type=int, nargs='?',default=50,help='patience')     # 在验证指标没有改善的情况下，提前停止训练所需的历元数。当性能趋于平稳时停止训练，有助于防止过度拟合。
    parser.add_argument('exist_ok', type=bool, nargs='?',default=False,help='exist_ok')    # 如果为 True，则允许覆盖现有的项目/名称目录。这对迭代实验非常有用，无需手动清除之前的输出。
    parser.add_argument('single_cls', type=bool,nargs='?', default=False,help='single_cls')      # 在训练过程中将多类数据集中的所有类别视为单一类别。适用于二元分类任务，或侧重于对象的存在而非分类。
    parser.add_argument('imgsz', type=int,nargs='?', default=640,help='imgsz')   # 用于训练的目标图像尺寸。所有图像在输入模型前都会被调整到这一尺寸。影响模型精度和计算复杂度。


    # 解析命令行参数
    args = parser.parse_args()
    print("参数解析完成")
    current_directory = os.getcwd()
    project_path = os.path.join(current_directory,'projects', args.project_name)
    yaml_path = os.path.join(project_path, 'train.yaml')
    result_path = os.path.join(project_path, 'train')
    # Load a model
    current_directory = os.getcwd()
    seg_filepath = os.path.join(current_directory, 'pre_model','yolov8n-seg.pt')
    model = YOLO(seg_filepath)  # load a pretrained model (recommended for training)
    # Train the model
    print("开始训练...")
    try:
        results = model.train(
            name=args.name,
            data=yaml_path,
            project=result_path,
            epochs=args.epochs,
            batch=args.batch,
            patience=args.patience,
            exist_ok=args.exist_ok,
            single_cls=args.single_cls,
            imgsz=args.imgsz,
            plots=True
        )
        print("训练完成")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    freeze_support()

    p = Process(target=train)
    p.start()
    p.join()