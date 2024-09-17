import subprocess
import flet as ft
import function
import os
import cv2
import threading
import time
import base64
from io import BytesIO
from PIL import Image
import shutil
from typing import Dict
import yolov8_train
import pandas as pd

'''TODO
label-studio 项目启动 命令优化

'''

class FAHAI:
    def __init__(self, page: ft.Page):
        self.page = page
        self.selected_project = None
        self.images_count = None
        self.labels_count = None
        self.classfile_exist = None
        self.camera_thread_instance = None
        self.prog_bars: Dict[str, ft.ProgressRing] = {}
        self.files = ft.Ref[ft.Column]()
        self.target_directory = None
        self.cap = None
        self.model_path = None
        self.setup_ui()

    def setup_ui(self):

        # components for projects_page
        self.selected_project_text = ft.Text(f"这是 {self.selected_project}")
        self.new_project_name = ft.TextField(label='Project Name', hint_text="Create a new project ?", expand=True)
        self.new_project_type = ft.Dropdown(
            width=200,
            label='Task Type',
            hint_text='Select task type',
            options=[
                ft.dropdown.Option("Detect"),
                ft.dropdown.Option("Segment"),
                ft.dropdown.Option("Classify",disabled=True),
                ft.dropdown.Option("Pose",disabled=True),
                ft.dropdown.Option("OBB",disabled=True),
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

        # components for datasets_page
        self.bg_img=os.path.join(os.getcwd(),'component','bosch-company-equipment-logo-wallpaper.jpg')
        self.img_element = ft.Image(src=self.bg_img, fit=ft.ImageFit.COVER,
                                    expand=True)
        self.text_element = ft.Text("Press START to start camera")
        self.camera_dropdown = ft.Dropdown(
            label="select CAM",
            options=[ft.dropdown.Option(str(i)) for i in range(5)], value="0",
            # height=50,
            width=200
        )
        self.start_button = ft.ElevatedButton("START", icon=ft.icons.PLAY_ARROW_ROUNDED, on_click=self.start_camera,expand=True)
        self.stop_button = ft.ElevatedButton("STOP", icon=ft.icons.STOP_ROUNDED, on_click=self.stop_camera,expand=True)
        self.take_photo_button = ft.ElevatedButton('Take Photo', icon=ft.icons.CAMERA, bgcolor='green',expand=True,
                                                   on_click=self.take_photo)
        self.predict_on = ft.Switch(label='Load YOLO', label_position=ft.LabelPosition.LEFT)
        self.upload_zip_button = ft.ElevatedButton("Auto-Upload", icon=ft.icons.UPLOAD, on_click=self.upload_zip,
                                                   expand=True)
        self.label_studio_button = ft.TextButton("Label-studio", icon=ft.icons.OPEN_IN_NEW,
                                                 on_click=self.open_label_studio)
        self.file_picker = ft.FilePicker(on_result=self.file_picker_result)
        self.page.overlay.append(self.file_picker)

        self.images_card = ft.Container(self.dataset_card(ft.icons.IMAGE, 'images', self.images_count), expand=3)
        self.labels_card = ft.Container(self.dataset_card(ft.icons.DOCUMENT_SCANNER, 'labels', self.labels_count),
                                        expand=3)
        self.classfile_card = ft.Container(
            self.dataset_card(ft.icons.DOCUMENT_SCANNER_OUTLINED, 'classes.txt', self.classfile_exist), expand=3)
        self.frame_width_input = ft.TextField(label='width', value="640", width=80,expand=True)
        self.frame_height_input = ft.TextField(label='height', value="480", width=80,expand=True)
        self.webview = ft.WebView(url="http://localhost:8088/projects/?page=1")
        self.datasets_page_porgress_ring = ft.ProgressRing(width=20, height=20, visible=False)
        self.datasets_page_labelstudio_ring = ft.ProgressRing(width=20, height=20, visible=False)

        # components for train_page
        self.train_settings_text = ft.Text("Train settings")

        self.train_settings_history = ft.Dropdown(on_change=self.update_result_table, expand=True)
        self.train_settings_resume = ft.TextButton("Resume Train", on_click=self.resume_train, expand=True,
                                                   disabled=True)
        self.train_settings_delete = ft.TextButton("Delete", icon=ft.icons.DELETE, on_click=self.delete_train,
                                                   expand=True, icon_color='red',
                                                   disabled=False)

        self.train_settings_exist_ok = ft.Switch(label='Exist OK', label_position=ft.LabelPosition.LEFT, value=True,
                                                 tooltip="-如果为 True，则允许覆盖现有的项目/名称目录。这对迭代实验非常有用，无需手动清除之前的输出-")
        self.train_settings_single_cls = ft.Switch(label='Single Class', label_position=ft.LabelPosition.LEFT,
                                                   tooltip="-在训练过程中将多类数据集中的所有类别视为单一类别。适用于二元分类任务，或侧重于对象的存在而非分类-")
        self.train_settings_train_name = ft.TextField(label='Train Name', hint_text="Train Name", expand=True,
                                                      value='train',
                                                      tooltip="-训练结果的存放文件夹名称-")
        self.train_settings_epochs = ft.TextField(label='Epochs', hint_text="Epochs", expand=True, value='50',
                                                  tooltip="-训练历元总数。每个历元代表对整个数据集进行一次完整的训练。调整该值会影响训练时间和模型性能-")
        self.train_settings_batch_size = ft.TextField(label='Batch Size', hint_text="Batch Size", value='2',
                                                      expand=True,
                                                      tooltip="-训练的批量大小，表示在更新模型内部参数之前要处理多少张图像。自动批处理 (batch=-1)会根据 GPU 内存可用性动态调整批处理大小-")
        self.train_settings_img_size_width = ft.TextField(label='width', hint_text="Image Size", value='640',
                                                    expand=True,
                                                    tooltip="-用于训练的目标图像尺寸。所有图像在输入模型前都会被调整到这一尺寸。影响模型精度和计算复杂度-")
        self.train_settings_img_size_height = ft.TextField(label='height', hint_text="Image Size", value='480',
                                                    expand=True,
                                                    tooltip="-用于训练的目标图像尺寸。所有图像在输入模型前都会被调整到这一尺寸。影响模型精度和计算复杂度-")


        self.train_settings_patience = ft.Slider(label='Patience', min=0, max=100, divisions=10, value=50, expand=True,
                                                 tooltip="-在验证指标没有改善的情况下，提前停止训练所需的历元数。当性能趋于平稳时停止训练，有助于防止过度拟合-")
        self.train_settings_degree = ft.Slider(label='Degree', min=0, max=360, divisions=10, value=20, expand=True,
                                               tooltip="-float -180 - +180  在指定的度数范围内随机旋转图像，提高模型识别不同方向物体的能力-")
        self.train_settings_translate = ft.Slider(label='Translate', min=0, max=1, divisions=10, value=0.1, expand=True,
                                                  tooltip="-float  0.0 - 1.0	以图像大小的一小部分水平和垂直平移图像，帮助学习检测部分可见的物体-")
        self.train_settings_scale = ft.Slider(label='Scale', min=0, max=1, divisions=10, value=0.5, expand=True,
                                              tooltip="-float 0.0 - 1.0  通过增益因子缩放图像，模拟物体与摄像机的不同距离-")
        self.train_settings_flipud = ft.Slider(label='Flipud', min=0, max=1, divisions=10, value=0, expand=True,
                                               tooltip="-float  0.0 - 1.0 以指定的概率将图像翻转过来，在不影响物体特征的情况下增加数据的可变性-")
        self.train_settings_fliplr = ft.Slider(label='Fliplr', min=0, max=1, divisions=10, value=0.5, expand=True,
                                               tooltip='float 0.0 - 1.0  以指定的概率将图像从左到右翻转，这对学习对称物体和增加数据集多样性非常有用')
        self.train_settings_erasing = ft.Slider(label='Erasing', min=0, max=1, divisions=10, value=0.3, expand=True,
                                                tooltip='Erasing')
        self.train_settings_mosaic = ft.Slider(label='Mosaic', min=0, max=1, divisions=10, value=0.0, expand=True,
                                               tooltip='float  0.0 - 1.0将四幅训练图像合成一幅，模拟不同的场景构成和物体互动。对复杂场景的理解非常有效')
        self.train_settings_mixup = ft.Slider(label='Mixup', min=0, max=1, divisions=10, value=0.0, expand=True,
                                              tooltip='float 0.0 - 1.0  混合两幅图像及其标签，创建合成图像。通过引入标签噪声和视觉变化，增强模型的泛化能力')
        self.train_settings_copy_paste = ft.Slider(label='Copy Paste', min=0, max=1, divisions=10, value=0.0,
                                                   expand=True,
                                                   tooltip='float 0.0 - 1.0  从一幅图像中复制物体并粘贴到另一幅图像上，用于增加物体实例和学习物体遮挡')
        self.train_settings_start_button = ft.ElevatedButton("Start Train", on_click=self.start_train, expand=True)
        self.train_settings_progress_ring = ft.ProgressBar(height=10, visible=False, expand=True)

        # 添加components 用于显示训练过程后台的日志进度
        self.train_progress_bar = ft.ProgressBar(visible=False, height=5)
        self.train_result_table = ft.Markdown(
            selectable=True,
            extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
            on_tap_link=lambda e: self.page.launch_url(e.data),
            code_theme="atom-one-dark",
            code_style=ft.TextStyle(font_family="Roboto Mono"),

        )

        # components for validate_page
        self.validate_settings_text = ft.Text("Validate settings")

        self.validate_settings_history = ft.Dropdown(on_change=self.find_weights, expand=True)
        self.validate_settings_delete = ft.TextButton("Delete", icon=ft.icons.DELETE, on_click=self.delete_train,
                                                      expand=True, icon_color='red',
                                                      disabled=False)
        self.validate_settings_weight = ft.Dropdown(expand=True, on_change=self.find_weight_path)
        self.validate_img_element = ft.Image(src=self.bg_img,
                                             fit=ft.ImageFit.COVER,
                                             expand=True)
        self.validate_camera_dropdown = ft.Dropdown(
            label="select CAM",
            options=[ft.dropdown.Option(str(i)) for i in range(5)], value="0",
            # height=50,
            width=200, expand=True
        )
        self.validate_weight_manual_select = ft.TextButton("Import", icon=ft.icons.DRIVE_FOLDER_UPLOAD,
                                                           on_click=lambda _: self.model_picker.pick_files(
                                                               allow_multiple=False, allowed_extensions=['pt']),
                                                           expand=True)
        self.validate_frame_width_input = ft.TextField(label='width', value="640", width=80, expand=True)
        self.validate_frame_height_input = ft.TextField(label='height', value="480", width=80, expand=True)
        self.validate_settings_start_button = ft.ElevatedButton("Start Validate", icon=ft.icons.PLAY_ARROW_ROUNDED,
                                                                on_click=self.start_validate_camera, expand=True)
        self.validate_settings_stop_button = ft.ElevatedButton("Stop Validate", icon=ft.icons.STOP_ROUNDED,
                                                               on_click=self.stop_validate_camera, expand=True)
        self.validate_settings_upload_button = ft.ElevatedButton("Upload image for predict", icon=ft.icons.UPLOAD,
                                                                 on_click=lambda _: self.image_picker.pick_files(
                                                                     allow_multiple=False,
                                                                     allowed_extensions=['bmp', 'jpg', 'jpeg', 'png']),
                                                                 expand=True)
        self.validate_results = ft.Markdown(
            selectable=True,
            extension_set="gitHubWeb",
            code_theme="atom-one-dark",
            code_style=ft.TextStyle(font_family="Roboto Mono"),
            expand=True)
        self.validate_results_text = ft.Text('Results', expand=True)
        self.validate_model_path = ft.Text(self.model_path, expand=True)
        self.model_picker = ft.FilePicker(on_result=self.on_model_picked)
        self.page.overlay.append(self.model_picker)  # FilePicker 需要添加到 overlay
        self.image_picker = ft.FilePicker(on_result=self.upload_img_predict)
        self.page.overlay.append(self.image_picker)  # FilePicker 需要添加到 overlay
        self.validate_progress_bar = ft.ProgressBar(visible=False, height=5, expand=True)
        self.validate_settings_conf = ft.Slider(label='Confidence', min=0, max=1, divisions=10, value=0.3, expand=True)
        self.validate_settings_iou = ft.Slider(label='IOU', min=0, max=1, divisions=10, value=0.5, expand=True)

        # datasets_page
        self.datasets_page = ft.Row([
            ft.Container(
            ft.Column(
                [
                    self.images_card,
                    self.labels_card,
                    self.classfile_card,
                    ft.Container(self.upload_zip_button, expand=1),
                    ft.Container(ft.Row([self.label_studio_button, self.datasets_page_labelstudio_ring]), expand=1),
                    ft.Column(ref=self.files, visible=False)
                ]), expand=2),
            ft.Container(
                ft.Column([self.img_element, ], alignment=ft.MainAxisAlignment.SPACE_AROUND,
                          horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                # bgcolor=ft.colors.BLACK if self.page.theme_mode == ft.ThemeMode.DARK else ft.colors.BLUE_50,
                expand=8),
            ft.Card(
            ft.Container(
                ft.Column([

                    ft.Row([self.start_button, self.stop_button], expand=True),
                    ft.Row([self.datasets_page_porgress_ring, self.text_element], expand=True),
                    ft.Row([self.predict_on], expand=True),
                    ft.Row([self.camera_dropdown], expand=True),
                    ft.Row([self.frame_width_input, ft.Text("x", width=20), self.frame_height_input], expand=True),
                    ft.Row([self.take_photo_button], expand=True),
                ], alignment=ft.MainAxisAlignment.SPACE_AROUND),
                # bgcolor=ft.colors.BLACK if self.page.theme_mode == ft.ThemeMode.DARK else ft.colors.BLUE_50,
                ),
            expand=2),
        ])

        # train_page
        self.train_page = ft.Row([
            ft.Card(
            ft.Container(
                ft.Column([
                ft.Row([ft.Text(''), self.train_settings_text]),
                ft.Row([ft.Text('history', width=80), self.train_settings_history, self.train_settings_resume,
                        self.train_settings_delete]),
                ft.Row([ft.Text('new train', width=80), self.train_settings_train_name,ft.Text('',width=10)],),
                ft.Row([ft.Text('', width=10), self.train_settings_exist_ok,self.train_settings_single_cls,ft.Text('',width=10)]),
                # ft.Row([ft.Text('single cls', width=80), self.train_settings_single_cls,ft.Text('',width=10)]),
                ft.Row([ft.Text('epochs', width=80), self.train_settings_epochs,ft.Text('',width=10)]),
                ft.Row([ft.Text('batch size', width=80), self.train_settings_batch_size,ft.Text('',width=10)]),
                ft.Row([ft.Text('img size', width=80), self.train_settings_img_size_width,ft.Text('X',width=10),self.train_settings_img_size_height,ft.Text('',width=10)]),
                ft.Row([ft.Text('patience', width=80), self.train_settings_patience,ft.Text('',width=10)]),
                ft.Row([ft.Text('degree', width=80), self.train_settings_degree,ft.Text('',width=10)]),
                ft.Row([ft.Text('translate', width=80), self.train_settings_translate,ft.Text('',width=10)]),
                ft.Row([ft.Text('scale', width=80), self.train_settings_scale,ft.Text('',width=10)]),
                ft.Row([ft.Text('flipud', width=80), self.train_settings_flipud,ft.Text('',width=10)]),
                ft.Row([ft.Text('fliplr', width=80), self.train_settings_fliplr,ft.Text('',width=10)]),
                ft.Row([ft.Text('erasing', width=80), self.train_settings_erasing,ft.Text('',width=10)]),
                ft.Row([ft.Text('mosaic', width=80), self.train_settings_mosaic,ft.Text('',width=10)]),
                ft.Row([ft.Text('mixup', width=80), self.train_settings_mixup,ft.Text('',width=10)]),
                ft.Row([ft.Text('copy paste', width=80), self.train_settings_copy_paste,ft.Text('',width=10)]),
                ft.Row([self.train_settings_start_button,ft.Text('',width=10)]),
                ft.Row([self.train_settings_progress_ring])

            ], scroll=ft.ScrollMode.ALWAYS,tight=False)
                , padding=5),expand=3),
            ft.Card(
            ft.Container(ft.Column([
                ft.Text('Train Progress'),
                self.train_progress_bar,
                ft.Column([self.train_result_table, ], scroll=ft.ScrollMode.ALWAYS, expand=1)

            ])),
                # bgcolor=ft.colors.BLUE_50,
                expand=7),
        ])

        # validate_page
        self.validate_page = ft.Row([
            ft.Container(
                ft.Column([
                    ft.Card(
                        ft.Column([
                            self.validate_settings_text,
                            ft.Row([ft.Text('history', width=80), self.validate_settings_history,
                                    self.validate_settings_delete,ft.Text('',width=10)]),
                            ft.Row([ft.Text('weight', width=80), self.validate_settings_weight,
                                    self.validate_weight_manual_select,ft.Text('',width=10)]),
                            ft.Row([self.validate_model_path,ft.Text('',width=10)]),
                            ft.Row([ft.Text('select CAM', width=80), self.validate_camera_dropdown,ft.Text('',width=10)]),
                            ft.Row([ft.Text('imgsz', width=80), self.validate_frame_width_input, ft.Text('X'),
                                    self.validate_frame_height_input,ft.Text('',width=10)]),
                            ft.Row([ft.Text('Confidence', width=80), self.validate_settings_conf]),
                            ft.Row([ft.Text('IOU', width=80), self.validate_settings_iou]),
                        ])),
                    ft.Card(
                        ft.Column([
                            ft.Row([self.validate_settings_start_button, self.validate_settings_stop_button]),
                            ft.Row([self.validate_settings_upload_button]),
                            ft.Row([self.validate_progress_bar])
                        ])),
                    ft.Card(
                        ft.Column([
                            ft.Row([self.validate_results_text]),
                            ft.Row([self.validate_results, ])

                        ], scroll=ft.ScrollMode.ALWAYS), expand=True)
                ], scroll=ft.ScrollMode.ALWAYS)
                , expand=3),
            ft.Container(ft.Column([self.validate_img_element], alignment=ft.MainAxisAlignment.SPACE_AROUND,
                                   horizontal_alignment=ft.CrossAxisAlignment.CENTER), expand=7),
        ])

        # tabs
        self.t = ft.Tabs(selected_index=0, animation_duration=300, on_change=self.update_datasets_card, tabs=[
            ft.Tab(
                text='Projects',
                icon=ft.icons.TASK,
                content=ft.Column(
                    [ft.Row([self.new_project_name, self.new_project_type,
                             ft.FloatingActionButton(icon=ft.icons.ADD, on_click=self.create_project_box), ]),
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
            # ft.Tab(
            #     text='Label-studio',
            #     icon=ft.icons.LABEL,
            #     content=ft.Container(self.webview, expand=1)
            # )
        ], expand=1)

        self.setup_page()

    def setup_page(self):
        self.page.views.clear()
        self.page.views.append(
            ft.View(
                "/",
                [
                    ft.AppBar(
                        title=ft.Text('Welcome to FAHAI develop tool !',
                                      spans=[
                                          ft.TextSpan(
                                              f'            project :[{self.selected_project}] selected ',
                                              ft.TextStyle(weight=ft.FontWeight.BOLD,
                                                           color=ft.colors.INDIGO_800 if self.selected_project else ft.colors.RED_800),

                                          ), ],text_align=ft.TextAlign.RIGHT),

                        bgcolor=ft.colors.SURFACE_VARIANT),
                    self.create_develop_content(),
                ]
            )
        )
        self.page.update()

    def on_model_picked(self, e: ft.FilePickerResultEvent):
        if e.files:
            # 将选择的文件路径显示在文本组件中
            self.model_path = e.files[0].path
            self.validate_model_path.value = self.model_path
            self.validate_model_path.update()

    def find_weight_path(self, e):
        self.model_path = os.path.join(os.getcwd(), 'projects', self.selected_project, 'train',
                                       self.validate_settings_history.value, 'weights',
                                       self.validate_settings_weight.value)
        self.validate_model_path.value = self.model_path
        self.validate_model_path.update()

    def start_validate_camera(self, e):
        self.validate_progress_bar.visible = True
        self.validate_progress_bar.update()
        camera_index = int(self.validate_camera_dropdown.value)
        model_path = self.model_path
        predict_on = True
        if self.camera_thread_instance and self.camera_thread_instance.is_alive():
            self.snack_message('CAM is working now', 'red')
            self.page.update()
            self.validate_progress_bar.visible = False
            self.validate_progress_bar.update()
            return
        self.camera_thread_instance = threading.Thread(target=self.camera_thread, args=(
        camera_index, self.validate_img_element, predict_on, model_path))
        self.camera_thread_instance.do_run = True
        self.camera_thread_instance.start()

    def stop_validate_camera(self, e):
        if self.camera_thread_instance:
            self.camera_thread_instance.do_run = False
            self.camera_thread_instance.join()
            self.snack_message('CAM is stopped', 'green')
            self.validate_img_element.src_base64 = ""
            self.page.update()

    def upload_img_predict(self, e: ft.FilePickerResultEvent):
        self.validate_progress_bar.visible = True
        self.validate_progress_bar.update()
        from ultralytics import YOLO
        try:
            model = self.load_model(YOLO, self.model_path)
        except Exception as e:
            self.snack_message(f"Error starting train: {e}", 'red')
            self.validate_progress_bar.visible = False
            self.validate_progress_bar.update()
            return
        if e.files:
            img_path = e.files[0].path
            conf = float(self.validate_settings_conf.value)
            iou = float(self.validate_settings_iou.value)
            width = int(self.validate_frame_width_input.value)
            height = int(self.validate_frame_height_input.value)
            try:
                res = model.predict(img_path, conf=conf, iou=iou, imgsz=(width, height))

                res_json = res[0].tojson()
                print(type(res_json))
                # formatted_json = json.dumps('```dart\n'+res_json+'\n```', indent=4,ensure_ascii=False)
                markdown_text = f"```dart\n{res_json}\n```"
                self.validate_results.value = markdown_text

                res_plotted = res[0].plot()
                frame_bgr = cv2.cvtColor(res_plotted, cv2.COLOR_RGB2BGR)
                img_pil = Image.fromarray(frame_bgr)
                img_byte_arr = BytesIO()
                img_pil.save(img_byte_arr, format="JPEG")
                img_byte_arr = img_byte_arr.getvalue()
                img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')
                self.validate_img_element.src_base64 = img_base64
                self.page.update()
            except:
                self.snack_message(f"Error starting train: {e}", 'red')
        self.validate_progress_bar.visible = False
        self.validate_progress_bar.update()

    def find_weights(self, e):
        project = self.selected_project
        train_name = self.validate_settings_history.value
        if project is None or train_name is None:
            return
        try:
            weights = function.find_weights(project, train_name)
            self.validate_settings_weight.options = [ft.dropdown.Option(weight) for weight in weights]
            # self.validate_settings_weight.value = weights[0]
            self.validate_settings_weight.update()

        except Exception as e:
            self.snack_message(f"Error finding weights for {train_name}: {e}", 'red')

    def delete_train(self, e):
        project = self.selected_project
        train_name = self.train_settings_history.value
        if project is None or train_name is None:
            return
        try:
            messagge = function.delete_train(project, train_name)
            self.snack_message(messagge, 'green')
            self.update_result_table(e)
            self.find_train_history(self.selected_project)

        except Exception as e:
            self.snack_message(f"Error deleting train {train_name}: {e}", 'red')

    def update_result_table(self, e):
        project = self.selected_project
        train_name = self.train_settings_history.value
        if project is None or train_name is None:
            return
        try:
            df = pd.read_csv(f"./projects/{project}/train/{train_name}/results.csv")
            row_count = len(df)
            # 读取 YAML 文件并获取 epoch 的值
            epochs = function.get_epoch_value(f"./projects/{project}/train/{train_name}/args.yaml")
            # 将数据表格转换为Markdown格式
            markdown_table = df.to_markdown(index=False)
            self.train_progress_bar.value = row_count / epochs
            self.train_result_table.value = markdown_table

        except FileNotFoundError:
            self.train_result_table.value = "文件 result.csv 未找到"
            self.train_progress_bar.value = 0

        self.train_progress_bar.visible = True
        self.train_progress_bar.color = 'green' if self.train_progress_bar.value >= 1 else 'red'
        self.train_progress_bar.update()
        self.train_result_table.update()

    def resume_train(self, e):
        self.snack_message('Resume Train, this function is not ready', 'red')

    def update_train_progress(self, e):
        while self.train_settings_progress_ring.visible == True:
            time.sleep(10)
            self.train_settings_history.value = self.train_settings_train_name.value
            self.train_settings_history.update()
            self.update_result_table(e)
            self.page.update()
        return

    def start_train(self, e):
        self.train_settings_progress_ring.visible = True
        updateThread = threading.Thread(target=self.update_train_progress, args=(e,))
        updateThread.start()
        if self.selected_project is None:
            self.train_settings_progress_ring.visible = False
            self.snack_message('Please select a project first', 'red')
            return
        function.create_yaml(self.selected_project)
        try:
            function.update_yaml(self.selected_project)
        except Exception as e:
            self.train_settings_progress_ring.visible = False
            self.snack_message(f"Error updating yaml: {e}", 'red')
            return
        self.snack_message('Train configure .yaml is ready, Start Train...', 'green')
        self.page.update()
        train_type = self.selected_project.split('_')[-1]
        project = str(self.selected_project)
        name = self.train_settings_train_name.value
        project_name = project
        epochs = int(self.train_settings_epochs.value)
        batch = int(self.train_settings_batch_size.value)
        patience = int(self.train_settings_patience.value)
        exist_ok = self.train_settings_exist_ok.value
        single_cls = self.train_settings_single_cls.value
        imgsz = (int(self.train_settings_img_size_width.value),int(self.train_settings_img_size_height.value))
        degrees = float(self.train_settings_degree.value)
        translate = float(self.train_settings_translate.value)
        scale = float(self.train_settings_scale.value)
        flipud = float(self.train_settings_flipud.value)
        fliplr = float(self.train_settings_fliplr.value)
        mosaic = float(self.train_settings_mosaic.value)
        mixup = float(self.train_settings_mixup.value)
        copy_paste = float(self.train_settings_copy_paste.value)
        if train_type == "Detect":
            try:
                yolov8_train.det_train(
                    name=name,
                    project_name=project_name,
                    epochs=epochs,
                    batch=batch,
                    patience=patience,
                    exist_ok=exist_ok,
                    single_cls=single_cls,
                    imgsz=imgsz,
                    degrees=degrees,
                    translate=translate,
                    scale=scale,
                    flipud=flipud,
                    fliplr=fliplr,
                    mosaic=mosaic,
                    mixup=mixup,
                    copy_paste=copy_paste
                )
            except Exception as e:
                self.snack_message(f"Error starting train: {e}", 'red')
        elif train_type == "Segment":
            try:
                yolov8_train.seg_train(
                    name=name,
                    project_name=project_name,
                    epochs=epochs,
                    batch=batch,
                    patience=patience,
                    exist_ok=exist_ok,
                    single_cls=single_cls,
                    imgsz=imgsz,
                    degrees=degrees,
                    translate=translate,
                    scale=scale,
                    flipud=flipud,
                    fliplr=fliplr,
                    mosaic=mosaic,
                    mixup=mixup,
                    copy_paste=copy_paste
                )
            except Exception as e:
                self.snack_message(f"Error starting train: {e}", 'red')
        else:
            self.snack_message("current only support Detect & Segment task", color='red')
        self.train_settings_progress_ring.visible = False
        self.page.update()

    def create_develop_content(self):
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
                try:
                    target_path = self.target_directory if f.name == 'classes.txt' else os.path.join(
                        self.target_directory, f.name)
                    shutil.copy(f.path, target_path)
                    self.snack_message(f"{f.name} 已复制到 {self.target_directory}", 'green')
                    self.prog_bars[f.name].value = 1.0
                    self.prog_bars[f.name].update()
                except Exception as ex:
                    self.snack_message(f"复制文件 {f.name} 失败: {ex}", 'red')
        self.count_datasets(self.selected_project)
        self.update_datasets_card(e)

    def upload_zip(self, e):
        self.snack_message('upload zip, this function is not ready', 'red')

    def upload_datasets(self, e):
        type = e.control.data
        self.target_directory = os.path.join(os.getcwd(), 'projects', self.selected_project, 'datasets', type)
        ext = ['png', 'jpg', 'jpeg', 'bmp'] if type == 'images' else ['txt']
        self.file_picker.pick_files(allow_multiple=True, allowed_extensions=ext)

    def delete_datasets(self, e):
        type = e.control.data
        target_path = os.path.join(os.getcwd(), 'projects', self.selected_project, 'datasets', type)
        message = function.delete_file(target_path)

        self.count_datasets(self.selected_project)
        self.update_datasets_card(e)
        self.snack_message(message, 'green')

    def open_label_studio(self, e):
        # 启动 Label Studio，指定端口
        port = 8088
        # 构建命令字符串，激活conda环境并运行label-studio
        self.datasets_page_labelstudio_ring.visible = True
        label_studio_thread = threading.Thread(target=self.label_studio, args=(port,))
        label_studio_thread.start()

    def label_studio(self, port):
        try:
            # "label-studio init test_project"   # 初始化项目
            # "label-studio start test_project --sampling sequential" # 启动项目
            subprocess.run(['label-studio', 'start', '--port', str(port)], check=True)  # 启动label studio

            messagge = f'Load label Studio success at {port}'
            color = 'green'
        except subprocess.CalledProcessError as e:
            self.datasets_page_labelstudio_ring.visible = False
            messagge = f"Error starting Label Studio: {e}"
            color = 'red'
        except FileNotFoundError:
            self.datasets_page_labelstudio_ring.visible = False
            messagge = "Label Studio is not installed or not found in PATH."
            color = 'red'
        self.snack_message(messagge, color)

    def update_datasets_card(self, e):
        self.images_card.content = self.dataset_card(ft.icons.IMAGE, 'images', self.images_count)
        self.labels_card.content = self.dataset_card(ft.icons.DOCUMENT_SCANNER, 'labels', self.labels_count)
        self.classfile_card.content = self.dataset_card(ft.icons.DOCUMENT_SCANNER_OUTLINED, 'classes.txt',
                                                        self.classfile_exist)
        self.images_card.update()
        self.labels_card.update()
        self.classfile_card.update()
        self.datasets_page.update()

    def dataset_card(self, icon, text, description, size=(200, 2)):
        description = str(description)
        return ft.Card(
            content=ft.Container(
                content=ft.Column(
                    [
                        ft.ListTile(
                            leading=ft.Icon(icon),
                            title=ft.Text(text),
                            subtitle=ft.Text(description, size=50),
                        ),
                        ft.Row(
                            [
                                ft.TextButton('import', icon=ft.icons.UPLOAD, data=text, on_click=self.upload_datasets),
                                ft.TextButton('delete all', icon=ft.icons.DELETE, data=text,
                                              on_click=self.delete_datasets)
                            ],
                            alignment=ft.MainAxisAlignment.END,
                        ),
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN
                ),
                expand=True
            )
        )

    def take_photo(self, e):
        # 获取当前摄像头frame，保存到指定目录下，结合camera_thread
        try:
            if not self.selected_project:
                self.snack_message('Please select a project first', 'red')
                return
            if not self.cap.isOpened():
                self.snack_message(f'摄像头未启动 {e}', 'red')
                return
            ret, frame = self.cap.read()
            frame = cv2.resize(frame, (int(self.frame_width_input.value), int(self.frame_height_input.value)))
            img_path = os.path.join(os.getcwd(), 'projects', self.selected_project, 'datasets', 'images')
            img_name = function.get_uuid()
            img_name = os.path.join(img_path, img_name + '.jpg')
            cv2.imwrite(img_name, frame)
            self.snack_message(f'Photo saved: {img_name}', 'green')
            self.count_datasets(self.selected_project)
            self.update_datasets_card(e)
        except Exception as e:
            self.snack_message(f' {e}', 'red')

    def load_model(self, YOLO, model_path):
        if model_path == None:
            model_path = os.path.join(os.getcwd(), 'pre_model', 'yolov8n.pt')
        print(model_path)
        model = YOLO(model_path)
        return model

    def camera_thread(self, camera_index, img_element, predict_on=False, model_path=None):
        if predict_on:
            self.datasets_page_porgress_ring.visible = True
            self.validate_progress_bar.visible = True
            self.validate_progress_bar.update()
            self.text_element.value = "Loading model... "
            self.page.update()
            from ultralytics import YOLO
            model = self.load_model(YOLO, model_path)
            self.text_element.value = "Load model success"
            self.datasets_page_porgress_ring.visible = False
            self.validate_progress_bar.visible = False
            self.validate_progress_bar.update()
            self.page.update()

        self.text_element.value = "Loading CAM... "
        self.datasets_page_porgress_ring.visible = True
        self.validate_progress_bar.visible = True
        self.validate_progress_bar.update()
        self.page.update()

        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            self.text_element.value = "CAM start failed"
            self.snack_message("CAM start failed", 'red')
            self.datasets_page_porgress_ring.visible = False
            self.validate_progress_bar.visible = False
            self.validate_progress_bar.update()
            self.page.update()
            return
        self.text_element.value = "CAM start success"
        self.snack_message("CAM start success", 'green')
        self.datasets_page_porgress_ring.visible = False
        self.validate_progress_bar.visible = False
        self.validate_progress_bar.update()
        self.page.update()
        while getattr(threading.currentThread(), "do_run", True):
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            conf = float(self.validate_settings_conf.value)
            iou = float(self.validate_settings_iou.value)
            width = int(self.validate_frame_width_input.value)
            height = int(self.validate_frame_height_input.value)

            try:
                res = model.predict(frame, conf=conf, iou=iou, imgsz=(width, height))
                res_plotted = res[0].plot()
                res_json = res[0].tojson()
                print(type(res_json))
                # formatted_json = json.dumps('```dart\n'+res_json+'\n```', indent=4,ensure_ascii=False)
                markdown_text = f"```dart\n{res_json}\n```"
                self.validate_results.value = markdown_text
            except:
                res_plotted = frame
            img_pil = Image.fromarray(res_plotted)
            img_byte_arr = BytesIO()
            img_pil.save(img_byte_arr, format="JPEG")
            img_byte_arr = img_byte_arr.getvalue()
            img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')
            img_element.src_base64 = img_base64
            self.page.update()
            time.sleep(0.03)
        self.cap.release()

    def start_camera(self, e):
        camera_index = int(self.camera_dropdown.value)
        predict_on = self.predict_on.value
        if self.camera_thread_instance and self.camera_thread_instance.is_alive():
            self.text_element.value = "摄像头已经在运行"
            self.page.update()
            return
        self.camera_thread_instance = threading.Thread(target=self.camera_thread,
                                                       args=(camera_index, self.img_element, predict_on))
        self.camera_thread_instance.do_run = True
        self.camera_thread_instance.start()

    def stop_camera(self, e):
        if self.camera_thread_instance:
            self.camera_thread_instance.do_run = False
            self.camera_thread_instance.join()
            self.text_element.value = "摄像头已停止"
            self.img_element.src_base64 = ""
            self.page.update()

    def snack_message(self, message, color):
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
            os.path.join(os.getcwd(), 'projects', str(selected_project), 'datasets', 'labels')) or function.find_file(
            os.path.join(os.getcwd(), 'projects', str(selected_project), 'datasets'))
        print(self.images_count, self.labels_count, self.classfile_exist)
        self.page.update()

    def find_train_history(self, project):
        selected_project = project
        self.train_settings_history.options.clear()
        self.validate_settings_history.options.clear()
        train_folder = os.path.join(os.getcwd(), 'projects', selected_project, 'train')
        if not os.path.exists(train_folder):
            return
        for root, dirs, files in os.walk(train_folder):
            # 找到train_folder下的所有文件夹
            for dir in dirs:
                if dir != 'weights':
                    self.train_settings_history.options.append(ft.dropdown.Option(dir))
                    self.validate_settings_history.options.append(ft.dropdown.Option(dir))

        self.page.update()

    def entry_project(self, e):
        self.selected_project = e.control.data
        self.count_datasets(self.selected_project)
        self.find_train_history(self.selected_project)
        self.page.update()
        self.setup_page()

    def open_project_folder(self,e):
        project = e.control.data
        project_path = os.path.join(os.getcwd(), 'projects', project)

        # 判断当前操作系统
        import platform
        current_os = platform.system()
        print(f"当前操作系统: {current_os}")
        # os.system(f'explorer {project_path}')# windows
        if current_os == 'Darwin':
            os.system(f'open {project_path}')# mac Darwin
        else:
            os.system(f'explorer {project_path}')# windows

        self.snack_message(f'Open project folder: {project_path}', 'green')

    def delete_project(self, e):
        card_title = e.control.data
        message = function.delete_project(card_title)
        self.make_projects_gridview()
        self.snack_message(message, 'green')

    def project_card(self, icon, text, description, action1, action2, size=(200, 2)):
        return ft.Card(
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
                                ft.TextButton('Folder',icon=ft.icons.FOLDER,data=text,on_click=self.open_project_folder),
                                # ft.TextButton(action1, icon=ft.icons.SETTINGS, data=text, on_click=self.entry_project),
                                ft.TextButton(action2, icon=ft.icons.DELETE, data=text, on_click=self.delete_project)
                            ],
                            alignment=ft.MainAxisAlignment.END,
                        ),
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN
                ),
                width=size[0],
                height=size[0],
                padding=size[1],
                data=text,
                on_click=self.entry_project
            )
        )

    def create_project_box(self, e):
        result_message = function.creat_project(self.new_project_name.value, self.new_project_type.value)
        self.new_project_name.value = ""
        self.new_project_type.value = None
        self.snack_message(result_message, 'green')
        self.make_projects_gridview()
        self.page.update()

    def make_projects_gridview(self):
        projects_list = function.get_project_folders()
        self.images.controls.clear()
        for p in projects_list:
            p_path = os.path.join(os.getcwd(), 'projects', p)
            create_time, modify_time, _ = function.get_folder_info(p_path)
            description = f'create time :\n{create_time}\nmodify time:\n{modify_time}\n'
            self.images.controls.append(
                self.project_card(icon=ft.icons.AUTO_AWESOME, text=p, description=description, action1='Select',
                                  action2='Delete')
            )
        return self.images


def main(page: ft.Page):
    page.title = "BOSCH_HzP_AI"
    page.padding = 0
    page.theme = ft.theme.Theme(font_family="Verdana", color_scheme_seed='blue')
    page.theme.page_transitions.windows = "cupertino"

    page.fonts = {
        "Roboto Mono": "RobotoMono-VariableFont_wght.ttf",
    }
    # page.fonts = {"Pacifico": "Pacifico-Regular.ttf"}
    page.bgcolor = ft.colors.BLUE_GREY_200
    page.window_maximized = True

    app = FAHAI(page)


if __name__ == "__main__":
    ft.app(target=main)
