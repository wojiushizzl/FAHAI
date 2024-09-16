import flet as ft
import function
import os
import cv2
import threading
import time
import base64
from io import BytesIO
from PIL import Image
import subprocess
from typing import Dict
import shutil


class FAHAI():
    def __init__(self, page: ft.Page):
        super().__init__()
        self.page = page

        self.selected_project = None

        self.selected_project_text = ft.Text(f"这是 {self.selected_project}")

        self.new_project_name = ft.TextField(label='Project Name', hint_text="Create a new project ?", expand=True)
        # ['Classify', 'Detect', 'Segment', 'Pose', 'OBB']
        self.new_project_type = ft.Dropdown(
            width=200,
            label='Task Type',
            hint_text='Select task type',
            options=[
                # ft.dropdown.Option("Classify"),
                ft.dropdown.Option("Detect"),
                ft.dropdown.Option("Segment"),
                # ft.dropdown.Option("Pose"),
                # ft.dropdown.Option("OBB"),
            ],
        )
        self.images = ft.GridView(
            expand=1,
            runs_count=5,
            max_extent=300,
            child_aspect_ratio=1.0,
            spacing=5,
            run_spacing=5,
        )

        # 添加一个显示摄像头视频流的Image控件
        self.img_element = ft.Image(src="./component/background.png", fit=ft.ImageFit.COVER, expand=True)

        # 添加一个显示文字的Text控件
        self.text_element = ft.Text("waiting...")

        # 摄像头选择的 Dropdown 控件
        self.camera_dropdown = ft.Dropdown(
            label="选择摄像头",
            options=[ft.dropdown.Option(str(i)) for i in range(5)], value="0",
            height=50,
            width=100
        )
        # 数据集采集按钮组
        self.start_button = ft.ElevatedButton("开始", on_click=self.start_camera)
        self.stop_button = ft.ElevatedButton("结束", on_click=self.stop_camera)
        self.take_photo_button = ft.ElevatedButton('Take Photo', icon=ft.icons.CAMERA, bgcolor='green',
                                                   on_click=self.take_photo)
        self.predict_on = ft.Switch(label='Load YOLO', label_position=ft.LabelPosition.LEFT)
        self.upload_zip_button = ft.ElevatedButton("Auto-Upload", icon=ft.icons.UPLOAD, on_click=self.upload_zip,
                                                   expand=True)
        self.label_studio_button = ft.TextButton("Label-studio", icon=ft.icons.OPEN_IN_NEW,
                                                 on_click=self.open_label_studio, )
        self.images_count = None
        self.labels_count = None
        self.classfile_exist = None
        # 摄像头线程的引用
        self.camera_thread_instance = None
        # 文件选择器
        self.file_picker = ft.FilePicker(on_result=self.file_picker_result)
        self.prog_bars: Dict[str, ft.ProgressRing] = {}
        self.files = ft.Ref[ft.Column]()
        self.target_directory = None
        self.page.overlay.append(self.file_picker)

        self.images_card = ft.Container(self.dataset_card(ft.icons.IMAGE, 'images', self.images_count), expand=3)
        self.labels_card = ft.Container(self.dataset_card(ft.icons.DOCUMENT_SCANNER, 'labels', self.labels_count),
                                        expand=3)
        self.classfile_card = ft.Container(self.dataset_card(ft.icons.DOCUMENT_SCANNER_OUTLINED, 'classes.txt',
                                                             self.classfile_exist), expand=3)

        # 数据集页面
        self.datasets_page = ft.Row(
            [ft.Container(
                ft.Column(
                    [
                        self.images_card,
                        self.labels_card,
                        self.classfile_card,
                        ft.Container(self.upload_zip_button, expand=1),
                        ft.Container(self.label_studio_button, expand=1),
                        ft.Column(ref=self.files, visible=False)
                    ]), expand=2),
                ft.Container(
                    ft.Column([self.img_element, ], alignment=ft.MainAxisAlignment.SPACE_AROUND,
                              horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                    bgcolor=ft.colors.BLUE_50, expand=8),
                ft.Container(

                    ft.Column([
                        ft.Row([self.start_button, self.stop_button], expand=True),
                        ft.Row([self.predict_on], expand=True),
                        ft.Row([self.camera_dropdown], expand=True),
                        ft.Row([self.text_element], expand=True),
                        ft.Row(
                            [ft.TextField(label='width', value="640", width=80), ft.Text("x", width=20),
                             ft.TextField(label='height', value="480", width=80)], expand=True),
                        ft.Row([self.take_photo_button], expand=True),
                    ], alignment=ft.MainAxisAlignment.SPACE_AROUND),
                    bgcolor=ft.colors.BLUE_50, expand=2),
            ]
        )
        self.train_page = ft.Container(self.selected_project_text)
        self.validate_page = ft.Container(self.selected_project_text)

        self.t = ft.Tabs(selected_index=0, animation_duration=300, on_change=self.update_datasets_card, tabs=[
            ft.Tab(
                text='Projects',
                icon=ft.icons.TASK,
                content=ft.Column(
                    [ft.Row([self.new_project_name, self.new_project_type,
                             ft.FloatingActionButton(icon=ft.icons.ADD,
                                                     on_click=self.create_project_box), ]),
                     self.images, ])
            ),
            ft.Tab(
                text="Datasets",
                icon=ft.icons.DATASET,
                content=self.datasets_page
            ),
            ft.Tab(
                text="Train",
                icon=ft.icons.ANALYTICS,
                content=self.train_page,
            ),
            ft.Tab(
                text="Validate",
                icon=ft.icons.FACT_CHECK,
                content=self.validate_page,
            ),
        ], expand=1)

        self.setup_page()

    def setup_page(self):
        # 清除所有视图
        self.page.views.clear()
        # 创建开发页面视图
        self.page.views.append(
            ft.View(
                "/",
                [
                    ft.AppBar(title=ft.Text(
                        f'Welcome to FAHAI develop tool ! project :[{self.selected_project}] selected '),
                        bgcolor=ft.colors.SURFACE_VARIANT),
                    self.create_develop_content(),
                ]
            )
        )
        self.page.update()

    def create_develop_content(self):
        # 创建开发页面内容的函数
        self.make_projects_gridview()
        self.page.update()

        return self.t

    def file_picker_result(self, e: ft.FilePickerResultEvent):
        self.prog_bars.clear()
        if self.files.current:
            self.files.current.controls.clear()
        if e.files is not None:
            for f in e.files:
                prog = ft.ProgressRing(value=0, bgcolor="#eeeeee", width=20, height=20)
                self.prog_bars[f.name] = prog
                self.files.current.controls.append(ft.Row([prog, ft.Text(f.name)]))
                self.page.update()
                # 开始复制文件
                try:
                    # 构建目标文件路径
                    print(f.name)
                    if f.name == 'classes.txt':
                        target_path = self.target_directory
                    else:
                        target_path = os.path.join(self.target_directory, f.name)
                    shutil.copy(f.path, target_path)  # 复制文件到目标文件夹
                    self.snack_message(f"{f.name} 已复制到 {self.target_directory}", 'green')

                    # 更新进度条为100%
                    self.prog_bars[f.name].value = 1.0
                    self.prog_bars[f.name].update()

                except Exception as ex:
                    self.snack_message(f"复制文件 {f.name} 失败: {ex}", 'red')
        self.count_datasets(self.selected_project)
        self.update_datasets_card(e)
        # self.page.update()

    def upload_zip(self, e):
        self.snack_message('upload zip', 'green')

    def upload_datasets(self, e):
        type = e.control.data
        self.target_directory = os.path.join(os.getcwd(), 'projects', self.selected_project, 'datasets', type)
        if type == 'images':
            ext = ['png', 'jpg', 'jpeg', 'bmp']
        else:
            ext = ['txt']

        self.file_picker.pick_files(allow_multiple=True, allowed_extensions=ext)

    def delete_datasets(self, e):
        type = e.control.data
        target_path = os.path.join(os.getcwd(), 'projects', self.selected_project, 'datasets', type)
        self.snack_message(f'delete {target_path} ', 'green')

    def open_label_studio(self, e):
        self.snack_message('open label-studio', 'green')

    def update_datasets_card(self, e):
        # Update images count
        self.images_card.content = self.dataset_card(ft.icons.IMAGE, 'images', self.images_count)
        # Update labels count
        self.labels_card.content = self.dataset_card(ft.icons.DOCUMENT_SCANNER, 'labels', self.labels_count)
        # Update class file exist check
        self.classfile_card.content = self.dataset_card(ft.icons.DOCUMENT_SCANNER_OUTLINED, 'classes.txt',
                                                        self.classfile_exist)

        self.images_card.update()
        self.labels_card.update()
        self.classfile_card.update()
        self.datasets_page.update()
        # self.setup_page()
        print('updated ')

    def dataset_card(self, icon, text, description, size=(200, 2)):
        description = str(description)
        return ft.Card(
            content=ft.Container(
                content=ft.Column(
                    [
                        ft.ListTile(
                            leading=ft.Icon(icon),
                            title=ft.Text(text),
                            subtitle=ft.Text(description,size=50),
                        ),
                        ft.Row(
                            [
                                ft.TextButton('import', icon=ft.icons.UPLOAD, data=text, on_click=self.upload_datasets),
                                ft.TextButton('delete all', icon=ft.icons.DELETE, data=text,
                                              on_click=self.delete_datasets, disabled=True)
                            ],
                            alignment=ft.MainAxisAlignment.END,
                        ),
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN
                ),
                expand=True
            )
        )

    def take_photo(self, e):
        self.snack_message(message='Take photo', color='green')

    def load_model(self, YOLO):
        # Load a model
        current_directory = os.getcwd()
        seg_filepath = os.path.join(current_directory, 'pre_model', 'yolov8n.pt')
        model = YOLO(seg_filepath)  # load a pretrained model (recommended for training)
        return model

    # 摄像头读取线程
    def camera_thread(self, camera_index, predict_on=False):
        self.text_element.value = "Loading model... "
        self.page.update()
        if predict_on == True:
            from ultralytics import YOLO
            model = self.load_model(YOLO)

        self.text_element.value = "Load model success"
        self.page.update()
        self.text_element.value = "Loading CAM... "
        self.page.update()

        cap = cv2.VideoCapture(camera_index)

        # 检查摄像头是否打开成功
        if not cap.isOpened():
            self.text_element.value = "CAM start failed"
            self.page.update()
            return
        # 摄像头成功启动后，更新文字
        self.text_element.value = "CAM start success"
        self.page.update()
        while getattr(threading.currentThread(), "do_run", True):
            ret, frame = cap.read()
            if not ret:
                break

            # 将图像从BGR转换为RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                res = model.predict(frame)
                res_plotted = res[0].plot()
            except:
                res_plotted = frame

            # 将OpenCV图像转换为PIL图像
            img_pil = Image.fromarray(res_plotted)

            # 将图像保存为字节流
            img_byte_arr = BytesIO()
            img_pil.save(img_byte_arr, format="JPEG")
            img_byte_arr = img_byte_arr.getvalue()

            # 将图像编码为base64字符串
            img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')

            # 更新Flet中的Image控件
            self.img_element.src_base64 = img_base64
            self.page.update()

            time.sleep(0.03)  # 控制帧率

        cap.release()

    # 开始摄像头的逻辑
    def start_camera(self, e):

        # 获取选择的摄像头编号
        camera_index = int(self.camera_dropdown.value)
        predict_on = self.predict_on.value
        # 如果已经有摄像头线程在运行，先终止它
        if self.camera_thread_instance and self.camera_thread_instance.is_alive():
            self.text_element.value = "摄像头已经在运行"
            self.page.update()
            return

        # 启动摄像头线程
        self.camera_thread_instance = threading.Thread(target=self.camera_thread, args=(camera_index, predict_on))
        self.camera_thread_instance.do_run = True
        self.camera_thread_instance.start()

    # 停止摄像头的逻辑
    def stop_camera(self, e):
        if self.camera_thread_instance:
            self.camera_thread_instance.do_run = False  # 终止线程
            self.camera_thread_instance.join()  # 等待线程结束
            self.text_element.value = "摄像头已停止"
            self.img_element.src_base64 = ""  # 清空图像
            self.page.update()

    def snack_message(self, message, color):
        # self.page.overlay.append(ft.SnackBar(ft.Text(message), bgcolor=color))
        self.page.snack_bar = ft.SnackBar(ft.Text(message), bgcolor=color)
        self.page.snack_bar.open = True
        self.page.update()

    def count_datasets(self, project):
        selected_project = project
        self.images_count = function.count_image_files(
            os.path.join(os.getcwd(), 'projects', str(selected_project), 'datasets', 'images'))
        self.labels_count = function.count_txt_files(
            os.path.join(os.getcwd(), 'projects', str(selected_project), 'datasets', 'labels'))
        self.classfile_exist = function.find_file(
            os.path.join(os.getcwd(), 'projects', str(selected_project), 'datasets',
                         'labels')) or function.find_file(
            os.path.join(os.getcwd(), 'projects', str(selected_project), 'datasets'))
        print(self.images_count, self.labels_count, self.classfile_exist)
        self.page.update()
        # self.setup_page()

    def entry_project(self, e):
        self.selected_project = e.control.data
        self.count_datasets(self.selected_project)
        # self.update_datasets_card()
        self.page.update()
        self.setup_page()

    def delete_project(self, e):
        print('delete ')
        card_title = e.control.data  # 获取存储在按钮中的卡片标识
        message = function.delete_project(card_title)
        self.make_projects_gridview()
        self.snack_message(message, 'green')

    def project_card(self, icon, text, description, action1, action2, size=(200, 2)):
        card = ft.Card(
            content=ft.Container(
                content=ft.Column(
                    [
                        ft.ListTile(
                            leading=ft.Icon(icon),
                            title=ft.Text(text),
                            subtitle=ft.Text(description),

                        ),
                        ft.Row(
                            [
                                ft.TextButton(action1, icon=ft.icons.SETTINGS, data=text, on_click=self.entry_project),
                                ft.TextButton(action2, icon=ft.icons.DELETE, data=text, on_click=self.delete_project)
                            ],
                            alignment=ft.MainAxisAlignment.END,
                        ),
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN
                ),
                width=size[0],
                height=size[0],
                padding=size[1],
            )
        )
        return card

    def create_project_box(self, e):
        result_message = function.creat_project(self.new_project_name.value, self.new_project_type.value)
        self.new_project_name.value = ""  # 清空文本框
        self.new_project_type.value = None  # 恢复默认（未选中状态）
        self.snack_message(result_message, 'green')
        self.make_projects_gridview()
        self.page.update()

    def make_projects_gridview(self):
        projects_list = function.get_project_folders()
        self.images.controls.clear()
        for i, p in enumerate(projects_list):
            p_path = os.path.join(os.getcwd(), 'projects', p)
            create_time, modify_time, tree_structure = function.get_folder_info(p_path)
            description = f'create time :\n{create_time}\nmodify time:\n {modify_time}\n'
            self.images.controls.append(
                self.project_card(icon=ft.icons.AUTO_AWESOME, text=p, description=description, action1='Select',
                                  action2='Delete')
            )
        return self.images


def main(page: ft.Page):
    page.title = "BOSCH_HzP_AI"
    page.padding = 0
    page.theme = ft.theme.Theme(font_family="Verdana")
    page.theme.page_transitions.windows = "cupertino"
    page.fonts = {"Pacifico": "Pacifico-Regular.ttf"}
    page.bgcolor = ft.colors.BLUE_GREY_200
    page.window.maximized = True

    # page.window.maximized
    app = FAHAI(page)


ft.app(target=main)
