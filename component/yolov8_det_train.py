from ultralytics import YOLO
from multiprocessing import Process, freeze_support
import os
import argparse

def train():
    # 创建解析器
    parser = argparse.ArgumentParser(description="Process some parameters.")

    # 添加参数
    parser.add_argument('name', type=str,  nargs='?',default='test',help='The train name')
    parser.add_argument('project_name', type=str,  nargs='?',default='test_Segment',help='The project_name')
    parser.add_argument('epochs', type=int,  nargs='?',default=50,help='epochs')     # 训练历元总数。每个历元代表对整个数据集进行一次完整的训练。调整该值会影响训练时间和模型性能。
    parser.add_argument('batch', type=int, nargs='?',default=2,help='batch')    # 训练的批量大小，表示在更新模型内部参数之前要处理多少张图像。自动批处理 (batch=-1)会根据 GPU 内存可用性动态调整批处理大小。
    parser.add_argument('patience', type=int, nargs='?',default=50,help='patience')     # 在验证指标没有改善的情况下，提前停止训练所需的历元数。当性能趋于平稳时停止训练，有助于防止过度拟合。
    parser.add_argument('exist_ok', type=bool, nargs='?',default=False,help='exist_ok')    # 如果为 True，则允许覆盖现有的项目/名称目录。这对迭代实验非常有用，无需手动清除之前的输出。
    parser.add_argument('single_cls', type=bool,nargs='?', default=False,help='single_cls')      # 在训练过程中将多类数据集中的所有类别视为单一类别。适用于二元分类任务，或侧重于对象的存在而非分类。
    parser.add_argument('imgsz', type=int,nargs='?', default=640,help='imgsz')   # 用于训练的目标图像尺寸。所有图像在输入模型前都会被调整到这一尺寸。影响模型精度和计算复杂度。

    parser.add_argument('degrees', type=int, nargs='?',default=20,help='degrees')     # float -180 - +180  在指定的度数范围内随机旋转图像，提高模型识别不同方向物体的能力。
    parser.add_argument('translate', type=float, nargs='?',default=0.1,help='translate')     # float  0.0 - 1.0	以图像大小的一小部分水平和垂直平移图像，帮助学习检测部分可见的物体。
    parser.add_argument('scale', type=float, nargs='?',default=0.5,help='scale')    # float 0.0 - 1.0  通过增益因子缩放图像，模拟物体与摄像机的不同距离。
    parser.add_argument('flipud', type=float, nargs='?',default=0,help='flipud')     # float  0.0 - 1.0 以指定的概率将图像翻转过来，在不影响物体特征的情况下增加数据的可变性。
    parser.add_argument('fliplr', type=float, nargs='?',default=0,help='fliplr e')     # float 0.0 - 1.0  以指定的概率将图像从左到右翻转，这对学习对称物体和增加数据集多样性非常有用。
    # parser.add_argument('erasing', type=float, nargs='?',default=0.4,help='erasing')    # float 0.0 - 1.0  在分类训练过程中随机擦除部分图像，鼓励模型将识别重点放在不明显的特征上。
    parser.add_argument('mosaic', type=float, nargs='?',default=1.0,help='mosaic')# float  0.0 - 1.0将四幅训练图像合成一幅，模拟不同的场景构成和物体互动。对复杂场景的理解非常有效。
    parser.add_argument('mixup', type=float, nargs='?',default=0,help='mixup')# float 0.0 - 1.0  混合两幅图像及其标签，创建合成图像。通过引入标签噪声和视觉变化，增强模型的泛化能力。
    parser.add_argument('copy_paste', type=float, nargs='?',default=0,help='copy_paste') # float 0.0 - 1.0  从一幅图像中复制物体并粘贴到另一幅图像上，用于增加物体实例和学习物体遮挡。
    # parser.add_argument('auto_augment', type=str, nargs='?',default='randaugment',help='auto_augment')    # 自动应用预定义的增强策略 (randaugment, autoaugment, augmix)，通过丰富视觉特征来优化分类任务。

    # 解析命令行参数
    args = parser.parse_args()
    print("参数解析完成")


    current_directory = os.getcwd()
    project_path = os.path.join(current_directory,'projects', args.project_name)
    yaml_path = os.path.join(project_path, 'train.yaml')
    result_path = os.path.join(project_path, 'train')
    # Load a model
    current_directory = os.getcwd()
    seg_filepath = os.path.join(current_directory, 'pre_model', 'yolov8n.pt')
    model = YOLO(seg_filepath)  # load a pretrained model (recommended for training)
    # Train the model
    print("开始训练...")
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
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        flipud=args.flipud,
        fliplr=args.fliplr,
        # erasing=args.erasing,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        # auto_augment=args.auto_augment,
        plots=True,
    )


if __name__ == "__main__":
    train()

    # freeze_support()
    #
    # p = Process(target=train)
    # p.start()
    # p.join()
