import os
import time
import shutil
import ruamel.yaml
import subprocess
import threading
import fnmatch
import uuid

yaml = ruamel.yaml.YAML()


def count_txt_files(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if fnmatch.fnmatch(file, "*.txt"):  # 匹配所有 .txt 文件
                count += 1
    return count


def count_image_files(directory):
    count = 0
    extensions = ["jpg", "png", "jpeg", "bmp", "gif"]
    for root, dirs, files in os.walk(directory):
        for file in files:
            for ext in extensions:
                if fnmatch.fnmatch(file, f"*.{ext}"):  # 匹配指定的图片文件类型
                    count += 1
    return count


def find_file(directory, filename='classes.txt'):
    for root, dirs, files in os.walk(directory):  # 遍历目录及其子目录
        if filename in files:  # 如果在当前文件夹中找到指定文件名
            return True
    return False



def get_project_folders():
    projects_path = os.path.join(os.getcwd(), 'projects')
    if os.path.exists(projects_path):
        return [f for f in os.listdir(projects_path) if os.path.isdir(os.path.join(projects_path, f))]
    return []


def get_folder_info(folder_path):
    # 获取文件夹的创建时间和修改时间
    create_time = time.ctime(os.path.getctime(folder_path))
    modify_time = time.ctime(os.path.getmtime(folder_path))

    # 获取文件夹的树结构
    tree_structure = []
    for root, dirs, files in os.walk(folder_path):
        level = root.replace(folder_path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        tree_structure.append(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            tree_structure.append(f"{subindent}{f}")
    return create_time, modify_time, tree_structure



# 删除文件或者文件夹路径，如果是文件夹则保留文件夹，删除文件夹下所有文件
def delete_file(file_path):
    try:
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            for f in os.listdir(file_path):
                os.remove(os.path.join(file_path, f))
        message = f"文件 '{file_path}' 已经删除。"
    except Exception as e:
        message = f"删除文件或文件夹时出错: {e}"
    return message


def get_uuid():
    return str(uuid.uuid1())


def creat_project(folder_name, task_type):
    projects_path = os.path.join(os.getcwd(), 'projects')
    if not os.path.exists(projects_path):
        os.makedirs(projects_path)
    new_folder_path = os.path.join(projects_path, folder_name + '_' + str(task_type))
    datasets_folder_path = os.path.join(new_folder_path, 'datasets')
    train_folder_path = os.path.join(new_folder_path, 'train')
    images_folder_path = os.path.join(datasets_folder_path, 'images')
    labels_folder_path = os.path.join(datasets_folder_path, 'labels')

    try:
        os.makedirs(new_folder_path)
        os.makedirs(datasets_folder_path)
        os.makedirs(train_folder_path)
        os.makedirs(images_folder_path)
        os.makedirs(labels_folder_path)

        message = f"Project '{folder_name}' 已成功创建在 '{projects_path}' 下。"
    except FileExistsError:
        message = f"Project '{folder_name}' 已经存在。"
    except Exception as e:
        message = f"创建Project 时出错: {e}"
    return message


def delete_project(selected_folder):
    folder_path = os.path.join(os.getcwd(), 'projects', selected_folder)
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for file in files:
            os.remove(os.path.join(root, file))
        for dir in dirs:
            os.rmdir(os.path.join(root, dir))
    os.rmdir(folder_path)
    message = f"项目 '{selected_folder}' 已经删除。"
    return message



def get_all_classes(project_name):
    classes_txt_path = os.path.join(os.getcwd(), 'projects', project_name, 'datasets', 'classes.txt')
    # print(classes_txt_path)
    result_dict = {}
    if os.path.exists(classes_txt_path):
        with open(classes_txt_path, 'r') as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                # print(line)
                result_dict[i] = line.strip()
        return result_dict
    else:
        return False


def update_yaml(project_name):
    file_name = os.path.join('projects', project_name, 'train.yaml')

    # 读取复制的文件并更改其中一行代码
    with open(file_name, "r", encoding="utf-8") as file:
        lines = yaml.load(file)

    classes = get_all_classes(project_name)
    if classes:
        if "names" in lines:
            lines["names"] = classes

        # 将更改后的内容写回文件
        with open(file_name, "w", encoding="utf-8") as file:
            yaml.dump(lines, file)
        # print('yaml file updated')
    else:
        raise FileNotFoundError("classes.txt 文件不存在")


def get_epoch_value(yaml_file):
    # 打开并读取 YAML 文件
    with open(yaml_file, 'r', encoding="utf-8") as file:
        try:
            config = yaml.load(file)  # 解析 YAML 文件内容
            # 获取 epoch 的值
            if 'epochs' in config:
                return config['epochs']
            else:
                raise ValueError("YAML 文件中没有找到 'epoch' 键")
        except Exception as e:
            ValueError(f"读取 YAML 文件时出错: {e}")


def create_yaml(project_name):
    # file_name = project_name + '/train.yaml'
    # 复制文件
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, 'projects', project_name, 'train.yaml')
    ori_yamel = os.path.join(current_directory, 'component', 'train.yaml')
    shutil.copyfile(ori_yamel, file_path)

    # 读取复制的文件并更改其中一行代码
    with open(file_path, "r", encoding="utf-8") as file:
        lines = yaml.load(file)

    path = os.path.join('projects', project_name, 'datasets')
    if "path" in lines:
        lines["path"] = path

    # 将更改后的内容写回文件
    with open(file_path, "w", encoding="utf-8") as file:
        yaml.dump(lines, file)

def find_weights(project_name,trian_name):
    weights_path = os.path.join(os.getcwd(), 'projects', project_name, 'train', trian_name,'weights')
    if os.path.exists(weights_path):
        items = os.listdir(weights_path)
        weights_list = [item for item in items if os.path.isfile(os.path.join(weights_path, item))]
    else:
        weights_list = []

    print(weights_list)
    return weights_list

def delete_train(project, train_name):
    folder_path = os.path.join(os.getcwd(), 'projects', project, 'train', train_name)
    if os.path.exists(folder_path):
        try:
            # 删除文件夹及其所有内容
            shutil.rmtree(folder_path)
            message = (f"文件夹 '{folder_path}' 已删除")
        except Exception as e:
            message = (f"删除文件夹时出错: {e}")
    else:
        message = (f"文件夹 '{folder_path}' 不存在")
    return message

