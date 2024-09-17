import os
import time
import shutil
import subprocess
import threading
import base64
from io import BytesIO
from PIL import Image
import cv2
import flet as ft
import function
import pandas as pd
from typing import Dict

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
        self.bg_img = os.path.join(os.getcwd(), 'component', 'bosch-company-equipment-logo-wallpaper.jpg')
        self.project_list = function.get_project_list()
        self.setup_ui()

    def setup_ui(self):
        self.theme_switch = ft.IconButton(ft.icons.WB_SUNNY_OUTLINED, on_click=self.change_theme)
        self.deploy_settings_text = ft.Text("deploy settings")
        self.deploy_project_dropdown = ft.Dropdown(options=[ft.dropdown.Option(str(p)) for p in self.project_list], expand=True, on_change=self.entry_project)
        self.deploy_project_folder_button = ft.TextButton("Open project folder", icon=ft.icons.FOLDER_OPEN, on_click=self.open_project_folder, expand=True)
        self.deploy_settings_history = ft.Dropdown(on_change=self.find_weights, expand=True)
        self.deploy_settings_weight = ft.Dropdown(expand=True, on_change=self.find_weight_path)
        self.deploy_img_element = ft.Image(src=self.bg_img, fit=ft.ImageFit.COVER, expand=True)
        self.deploy_camera_dropdown = ft.Dropdown(label="select CAM", options=[ft.dropdown.Option(str(i)) for i in range(5)], value="0", width=200, expand=True)
        self.deploy_weight_manual_select = ft.TextButton("Import", icon=ft.icons.DRIVE_FOLDER_UPLOAD, on_click=lambda _: self.model_picker.pick_files(allow_multiple=False, allowed_extensions=['pt']), expand=True)
        self.deploy_frame_width_input = ft.TextField(label='width', value="640", width=80, expand=True)
        self.deploy_frame_height_input = ft.TextField(label='height', value="480", width=80, expand=True)
        self.deploy_settings_start_button = ft.ElevatedButton("Start deploy", icon=ft.icons.PLAY_ARROW_ROUNDED, icon_color='green', on_click=self.start_deploy_camera, expand=True, height=50)
        self.deploy_settings_stop_button = ft.ElevatedButton("Stop deploy", icon=ft.icons.STOP_ROUNDED, icon_color='red', on_click=self.stop_deploy_camera, expand=True, height=50)
        self.deploy_settings_upload_button = ft.ElevatedButton("Upload image for predict", icon=ft.icons.UPLOAD, on_click=lambda _: self.image_picker.pick_files(allow_multiple=False, allowed_extensions=['bmp', 'jpg', 'jpeg', 'png']), expand=True)
        self.deploy_results = ft.Markdown(selectable=True, extension_set="gitHubWeb", code_theme="atom-one-dark", code_style=ft.TextStyle(font_family="Roboto Mono"), expand=True)
        self.deploy_results_text = ft.Text('Results', expand=True)
        self.deploy_model_path = ft.Text(self.model_path, expand=True)
        self.model_picker = ft.FilePicker(on_result=self.on_model_picked)
        self.page.overlay.append(self.model_picker)
        self.image_picker = ft.FilePicker(on_result=self.upload_img_predict)
        self.page.overlay.append(self.image_picker)
        self.deploy_progress_bar = ft.ProgressBar(visible=False, height=5, expand=True)
        self.deploy_settings_conf = ft.Slider(label='Confidence', min=0, max=1, divisions=10, value=0.3, expand=True)
        self.deploy_settings_iou = ft.Slider(label='IOU', min=0, max=1, divisions=10, value=0.5, expand=True)

        self.deploy_page = ft.Row([
            ft.Container(
                ft.Column([
                    ft.Card(ft.Column([ft.Row([self.deploy_settings_start_button, self.deploy_settings_stop_button]), ft.Row([self.deploy_progress_bar])])),
                    ft.Card(ft.Column([ft.Row([ft.Text('Project', width=80), self.deploy_project_dropdown, self.deploy_project_folder_button, ft.Text('', width=10)]), ft.Row([ft.Text('history', width=80), self.deploy_settings_history, ft.Text('', width=10)]), ft.Row([ft.Text('weight', width=80), self.deploy_settings_weight, self.deploy_weight_manual_select, ft.Text('', width=10)]), ft.Row([self.deploy_model_path, ft.Text('', width=10)])])),
                    ft.Card(ft.Column([ft.Row([self.deploy_results_text]), ft.Row([self.deploy_results])], scroll=ft.ScrollMode.ALWAYS), expand=True)
                ], scroll=ft.ScrollMode.ALWAYS), expand=3),
            ft.Container(ft.Column([self.deploy_img_element], alignment=ft.MainAxisAlignment.SPACE_AROUND, horizontal_alignment=ft.CrossAxisAlignment.CENTER), expand=7),
        ])

        self.settings_page = ft.Container(ft.Row([
            ft.Container(ft.Column([ft.Card(ft.Column([ft.Row([ft.Text('Input', expand=True)]), ft.Row([self.deploy_settings_upload_button]), ft.Row([ft.Text('select CAM', width=80), self.deploy_camera_dropdown, ft.Text('', width=10)])]), expand=1), ft.Card(ft.Column([ft.Row([ft.Text('Formula', expand=True)]), ft.Row([ft.Text('imgsz', width=80), self.deploy_frame_width_input, ft.Text('X'), self.deploy_frame_height_input, ft.Text('', width=10)]), ft.Row([ft.Text('Confidence', width=80), self.deploy_settings_conf]), ft.Row([ft.Text('IOU', width=80), self.deploy_settings_iou])]), expand=1)]), expand=1),
            ft.Container(ft.Column([ft.Card(ft.Column([ft.Row([ft.Text('Rule', expand=True)])]), expand=1), ft.Card(ft.Column([ft.Row([ft.Text('OUTPUT', expand=True)])]), expand=1)]), expand=1),
        ]))

        self.t = ft.Tabs(selected_index=0, animation_duration=300, tabs=[
            ft.Tab(text="deploy", icon=ft.icons.FACT_CHECK, content=self.deploy_page),
            ft.Tab(text='Settings', icon=ft.icons.SETTINGS, content=self.settings_page)
        ], expand=1)

        self.setup_page()

    def change_theme(self, e):
        self.page.theme_mode = ft.ThemeMode.DARK if self.page.theme_mode == ft.ThemeMode.LIGHT else ft.ThemeMode.LIGHT
        self.theme_switch.icon = ft.icons.WB_SUNNY_OUTLINED if self.page.theme_mode == ft.ThemeMode.LIGHT else ft.icons.NIGHTLIGHT_OUTLINED
        self.theme_switch.update()
        self.page.update()

    def setup_page(self):
        self.page.views.clear()
        self.page.views.append(ft.View("/", [ft.AppBar(leading=ft.Icon(ft.icons.CATCHING_POKEMON), adaptive=True, title=ft.Text('Welcome to FAHAI deploy tool !'), center_title=False, bgcolor=ft.colors.SURFACE_VARIANT, actions=[self.theme_switch, ft.IconButton(ft.icons.EXIT_TO_APP, on_click=lambda e: self.page.window_close())]), self.t]))
        self.page.update()

    def on_model_picked(self, e: ft.FilePickerResultEvent):
        if e.files:
            self.model_path = e.files[0].path
            self.deploy_model_path.value = self.model_path
            self.deploy_model_path.update()

    def find_weight_path(self, e):
        self.model_path = os.path.join(os.getcwd(), 'projects', self.selected_project, 'train', self.deploy_settings_history.value, 'weights', self.deploy_settings_weight.value)
        self.deploy_model_path.value = self.model_path
        self.deploy_model_path.update()

    def start_deploy_camera(self, e):
        self.deploy_progress_bar.visible = True
        self.deploy_progress_bar.update()
        camera_index = int(self.deploy_camera_dropdown.value)
        model_path = self.model_path
        predict_on = True
        if self.camera_thread_instance and self.camera_thread_instance.is_alive():
            self.snack_message('CAM is working now', 'red')
            self.page.update()
            self.deploy_progress_bar.visible = False
            self.deploy_progress_bar.update()
            return
        self.camera_thread_instance = threading.Thread(target=self.camera_thread, args=(camera_index, self.deploy_img_element, predict_on, model_path))
        self.camera_thread_instance.do_run = True
        self.camera_thread_instance.start()

    def stop_deploy_camera(self, e):
        if self.camera_thread_instance:
            self.camera_thread_instance.do_run = False
            self.camera_thread_instance.join()
            self.snack_message('CAM is stopped', 'green')
            self.deploy_img_element.src_base64 = ""
            self.page.update()

    def upload_img_predict(self, e: ft.FilePickerResultEvent):
        self.deploy_progress_bar.visible = True
        self.deploy_progress_bar.update()
        from ultralytics import YOLO
        try:
            model = self.load_model(YOLO, self.model_path)
        except Exception as e:
            self.snack_message(f"Error starting train: {e}", 'red')
            self.deploy_progress_bar.visible = False
            self.deploy_progress_bar.update()
            return
        if e.files:
            img_path = e.files[0].path
            conf = float(self.deploy_settings_conf.value)
            iou = float(self.deploy_settings_iou.value)
            width = int(self.deploy_frame_width_input.value)
            height = int(self.deploy_frame_height_input.value)
            try:
                res = model.predict(img_path, conf=conf, iou=iou, imgsz=(width, height))
                res_json = res[0].tojson()
                markdown_text = f"```dart\n{res_json}\n```"
                self.deploy_results.value = markdown_text
                res_plotted = res[0].plot()
                frame_bgr = cv2.cvtColor(res_plotted, cv2.COLOR_RGB2BGR)
                img_pil = Image.fromarray(frame_bgr)
                img_byte_arr = BytesIO()
                img_pil.save(img_byte_arr, format="JPEG")
                img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
                self.deploy_img_element.src_base64 = img_base64
                self.page.update()
            except:
                self.snack_message(f"Error starting train: {e}", 'red')
        self.deploy_progress_bar.visible = False
        self.deploy_progress_bar.update()

    def find_weights(self, e):
        project = self.selected_project
        train_name = self.deploy_settings_history.value
        if project is None or train_name is None:
            return
        try:
            weights = function.find_weights(project, train_name)
            self.deploy_settings_weight.options = [ft.dropdown.Option(weight) for weight in weights]
            self.deploy_settings_weight.update()
        except Exception as e:
            self.snack_message(f"Error finding weights for {train_name}: {e}", 'red')

    def load_model(self, YOLO, model_path):
        if model_path is None:
            model_path = os.path.join(os.getcwd(), 'pre_model', 'yolov8n.pt')
        return YOLO(model_path)

    def camera_thread(self, camera_index, img_element, predict_on=False, model_path=None):
        if predict_on:
            self.deploy_progress_bar.visible = True
            self.deploy_progress_bar.update()
            self.page.update()
            from ultralytics import YOLO
            model = self.load_model(YOLO, model_path)
            self.deploy_progress_bar.visible = False
            self.deploy_progress_bar.update()
            self.page.update()

        self.deploy_progress_bar.visible = True
        self.deploy_progress_bar.update()
        self.page.update()

        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            self.snack_message("CAM start failed", 'red')
            self.deploy_progress_bar.visible = False
            self.deploy_progress_bar.update()
            self.page.update()
            return
        self.snack_message("CAM start success", 'green')
        self.deploy_progress_bar.visible = False
        self.deploy_progress_bar.update()
        self.page.update()
        while getattr(threading.currentThread(), "do_run", True):
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            conf = float(self.deploy_settings_conf.value)
            iou = float(self.deploy_settings_iou.value)
            width = int(self.deploy_frame_width_input.value)
            height = int(self.deploy_frame_height_input.value)
            try:
                res = model.predict(frame, conf=conf, iou=iou, imgsz=(width, height))
                res_plotted = res[0].plot()
                res_json = res[0].tojson()
                markdown_text = f"```dart\n{res_json}\n```"
                self.deploy_results.value = markdown_text
            except:
                res_plotted = frame
            img_pil = Image.fromarray(res_plotted)
            img_byte_arr = BytesIO()
            img_pil.save(img_byte_arr, format="JPEG")
            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            img_element.src_base64 = img_base64
            self.page.update()
            time.sleep(0.03)
        self.cap.release()

    def snack_message(self, message, color):
        self.page.snack_bar = ft.SnackBar(ft.Text(message), bgcolor=color)
        self.page.snack_bar.open = True
        self.page.update()

    def find_train_history(self, project):
        self.deploy_settings_history.options.clear()
        train_folder = os.path.join(os.getcwd(), 'projects', project, 'train')
        if not os.path.exists(train_folder):
            return
        for dir in next(os.walk(train_folder))[1]:
            if dir != 'weights':
                self.deploy_settings_history.options.append(ft.dropdown.Option(dir))
        self.page.update()

    def entry_project(self, e):
        self.selected_project = self.deploy_project_dropdown.value
        self.find_train_history(self.selected_project)
        self.page.update()

    def open_project_folder(self, e):
        project = self.selected_project
        if project is None:
            self.snack_message('Please select a project', 'red')
            return
        project_path = os.path.join(os.getcwd(), 'projects', project)
        import platform
        current_os = platform.system()
        if current_os == 'Darwin':
            os.system(f'open {project_path}')
        else:
            os.system(f'explorer {project_path}')
        self.snack_message(f'Open project folder: {project_path}', 'green')

def main(page: ft.Page):
    page.title = "BOSCH_HzP_AI_Platform"
    page.padding = 0
    page.theme = ft.theme.Theme(font_family="Verdana", color_scheme_seed='blue')
    page.theme.page_transitions.windows = "cupertino"
    page.fonts = {"Roboto Mono": "RobotoMono-VariableFont_wght.ttf"}
    page.window_frameless = True
    page.window_resizable = False
    page.bgcolor = ft.colors.BLUE_GREY_200
    page.window_maximized = True
    app = FAHAI(page)

if __name__ == "__main__":
    ft.app(target=main)