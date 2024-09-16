from  ultralytics import  YOLO
from PIL import Image
import io


def load_model(selected_weight_path):
    model=YOLO(selected_weight_path)
    classes_dir = model.names
    classes_list = list(classes_dir.values())
    return model

def model_val(selected_weight_path):
    model=load_model(selected_weight_path)
    metrics =model.val()
    print(metrics)
    return metrics


def numpy_to_bytes(numpy_image,size):
    # 将 numpy 数组转换为 Pillow 图像
    img = Image.fromarray(numpy_image)

    # 如果指定了 size 参数，则调整图像大小
    if size:
        img = img.resize(size)

    # 将图像保存为字节流
    bio = io.BytesIO()
    img.save(bio, format="PNG")

    # 获取字节数据
    return bio.getvalue()

def image_predict_seg(image_path,selected_weight_path,size,conf=0.3,pre_imgsz=640):
    model=load_model(selected_weight_path)
    res=model.predict(image_path,conf=conf,imgsz=pre_imgsz)
    res_plotted = res[0].plot()[:, :, ::-1]
    image_bytes=numpy_to_bytes(res_plotted,size)
    return image_bytes