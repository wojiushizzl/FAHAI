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
import json
import sys
try:
    import Jetson.GPIO as GPIO
except:
    print("Jetson.GPIO not found")

class FAHAI:
    SETTINGS_FILE = 'settings.json'

    def __init__(self, page: ft.Page):
        self.page = page
        self.selected_project = None
        self.camera_thread_instance = None
        self.prog_bars = {}
        self.files = ft.Ref[ft.Column]()
        self.cap = None
        self.model_path = None
        # self.bg_img = os.path.join(os.getcwd(), 'component', 'bosch-company-equipment-logo-wallpaper.jpg')
        self.bg_img = os.path.join(os.getcwd(), 'component', 'siri.gif')
        self.classes = {}
        self.selected_classes = []
        self.project_list = function.get_project_list()
        self.RelayA = [21, 20, 26]
        self.setup_ui()
        self.initialize_gpio()
        self.load_settings()

    def setup_ui(self):
        self.theme_switch = ft.IconButton(ft.icons.WB_SUNNY_OUTLINED, on_click=self.change_theme)
        self.deploy_choice_text = ft.Text(expand=True, size=40, text_align=ft.TextAlign.CENTER)
        self.deploy_choice_dropdown = ft.Dropdown(label='select choice', expand=True, on_change=self.load_settings)
        self.deploy_add_choice_button = ft.IconButton(ft.icons.ADD, on_click=self.add_choice, width=60)
        self.deploy_save_choice_button = ft.IconButton(ft.icons.SAVE, on_click=self.save_settings, width=60)
        self.deploy_remove_choice_button = ft.IconButton(ft.icons.DELETE, on_click=self.remove_choice, width=60)
        self.deploy_project_dropdown = ft.Dropdown(options=[ft.dropdown.Option(str(p)) for p in self.project_list],
                                                   expand=True, on_change=self.entry_project)
        self.deploy_project_folder_button = ft.TextButton("Open project folder", icon=ft.icons.FOLDER_OPEN,
                                                          on_click=self.open_project_folder, expand=True)
        self.deploy_settings_history = ft.Dropdown(on_change=self.find_weights, expand=True)
        self.deploy_settings_weight = ft.Dropdown(expand=True, on_change=self.find_weight_path)
        self.deploy_img_element = ft.Image(src=self.bg_img, fit=ft.ImageFit.COVER, expand=True)
        self.deploy_camera_dropdown = ft.Dropdown(label="select CAM",
                                                  options=[ft.dropdown.Option(str(i)) for i in range(5)], value="0",
                                                  width=200, expand=True)
        self.deploy_weight_manual_select = ft.TextButton("Import .pt", icon=ft.icons.DRIVE_FOLDER_UPLOAD,
                                                         on_click=lambda _: self.model_picker.pick_files(
                                                             allow_multiple=False, allowed_extensions=['pt']),
                                                         expand=True)
        self.deploy_frame_width_input = ft.TextField(label='width', value="640", width=80, expand=True)
        self.deploy_frame_height_input = ft.TextField(label='height', value="480", width=80, expand=True)
        self.deploy_settings_start_button = ft.ElevatedButton("Start", icon=ft.icons.PLAY_ARROW_ROUNDED,
                                                              icon_color='green', on_click=self.start_deploy_camera,
                                                              expand=True, height=50)
        self.deploy_settings_stop_button = ft.ElevatedButton("Stop", icon=ft.icons.STOP_ROUNDED,
                                                             icon_color='red', on_click=self.stop_deploy_camera,
                                                             expand=True, height=50)
        self.deploy_settings_upload_button = ft.ElevatedButton("Upload image for predict", icon=ft.icons.UPLOAD,
                                                               on_click=lambda _: self.image_picker.pick_files(
                                                                   allow_multiple=False,
                                                                   allowed_extensions=['bmp', 'jpg', 'jpeg', 'png']),
                                                               expand=True)
        self.deploy_results = ft.Markdown(selectable=True, extension_set="gitHubWeb", code_theme="atom-one-dark",
                                          code_style=ft.TextStyle(font_family="Roboto Mono"), expand=True)
        self.deploy_results_text = ft.Text('Results', expand=True)
        self.deploy_model_path = ft.Text(self.model_path, italic=True, expand=True)
        self.model_picker = ft.FilePicker(on_result=self.on_model_picked)
        self.page.overlay.append(self.model_picker)
        self.image_picker = ft.FilePicker(on_result=self.upload_img_predict)
        self.page.overlay.append(self.image_picker)
        self.deploy_progress_bar = ft.ProgressBar(visible=False, height=5, expand=True)
        self.deploy_settings_conf = ft.Slider(label='Confidence', min=0, max=1, divisions=10, value=0.3, expand=True)
        self.deploy_settings_iou = ft.Slider(label='IOU', min=0, max=1, divisions=10, value=0.5, expand=True)
        self.deploy_input_trigger_type = ft.RadioGroup(
            content=ft.Column(
                [
                    ft.Radio(value="streaming", label="streaming", fill_color=ft.colors.GREEN),
                    ft.Radio(value="trigger", label="trigger", fill_color=ft.colors.GREEN),
                ]), disabled=True)

        self.deploy_input_CAM_type = ft.RadioGroup(
            content=ft.Column(
                [
                    ft.Radio(value="hikrobotic CAM", label="hikrobotic CAM", fill_color=ft.colors.GREEN),
                    ft.Radio(value="CV CAM", label="CV CAM", fill_color=ft.colors.GREEN),
                ]))

        self.deploy_logtic_class_select = ft.Column([])

        self.deploy_logtic_type = ft.RadioGroup(
            content=ft.Column(
                [
                    ft.Radio(value="detected results [include] left selected classes",
                             label="if detected results [include] left selected classes  -> true",
                             fill_color=ft.colors.GREEN),
                    ft.Radio(value="detected results [in] left selected classes",
                             label="if detected results [in] left selected classes -> true",
                             fill_color=ft.colors.GREEN),
                    ft.Radio(value="detected results & left selected classes [No intersection]",
                             label="if detected results & left selected classes [No intersection] -> true",
                             fill_color=ft.colors.GREEN),
                    ft.Radio(value="detected results & left selected classes [intersection]",
                             label="if detected results & left selected classes [intersection] -> true",
                             fill_color=ft.colors.GREEN),
                    ft.Radio(value="count",
                             label="count -> Number of detected objects",
                             fill_color=ft.colors.GREEN),

                ]))

        self.deploy_output_type = ft.RadioGroup(
            content=ft.Column(
                [
                    ft.Radio(value="Visualize", label="Visualize", fill_color=ft.colors.GREEN, toggleable=True),
                ]))
        self.deploy_output_GPIO = ft.RadioGroup(on_change=self.initialize_gpio,
            content=ft.Column(
                [
                    ft.Radio(value="GPIO_21", label="GPIO_21", fill_color=ft.colors.GREEN, toggleable=True),

                ]))

        self.deploy_page = ft.Row([
            ft.Container(
                ft.Column([
                    ft.Card(ft.Column([ft.Row([ft.Text('selected choice :'), self.deploy_choice_text])])),
                    ft.Card(ft.Column([ft.Row([self.deploy_settings_start_button, self.deploy_settings_stop_button]),
                                       ft.Row([self.deploy_progress_bar])])),
                    ft.Card(ft.Column([
                        # ft.Row([ft.Text('history', width=80), self.deploy_settings_history, ft.Text('', width=10)]),
                    ])),
                    ft.Card(ft.Column([ft.Row([self.deploy_results_text]), ft.Row([self.deploy_results])],
                                      scroll=ft.ScrollMode.ALWAYS), expand=True)
                ], scroll=ft.ScrollMode.ALWAYS), alignment=ft.alignment.top_left, expand=3),
            ft.Container(ft.Column([self.deploy_img_element], alignment=ft.MainAxisAlignment.SPACE_AROUND,
                                   horizontal_alignment=ft.CrossAxisAlignment.CENTER), expand=7),
        ])

        self.settings_page = ft.Column([
            ft.ExpansionTile(
                title=ft.Text("Choice Settings"),
                # subtitle=ft.Text("Deploy settings"),
                initially_expanded=True,
                controls=[
                    ft.Card(
                        ft.Column([
                            ft.Row([ft.Text('Choice', width=80), self.deploy_choice_dropdown,
                                    self.deploy_add_choice_button, self.deploy_save_choice_button,
                                    self.deploy_remove_choice_button, ft.Text('', width=10)]),

                            ft.ExpansionTile(
                                title=ft.Text("Project Settings"),
                                # subtitle=ft.Text("Deploy settings"),
                                initially_expanded=False,
                                controls=[
                                    ft.Card(
                                        ft.Column([
                                            ft.Row([ft.Text('Project', width=80), self.deploy_project_dropdown,
                                                    self.deploy_project_folder_button, ft.Text('', width=10)]),
                                            ft.Row([ft.Text('history', width=80), self.deploy_settings_history,
                                                    ft.Text('', width=10)]),
                                            ft.Row([ft.Text('weight', width=80), self.deploy_settings_weight,
                                                    self.deploy_weight_manual_select,
                                                    ft.Text('', width=10)]),
                                            ft.Row([ft.Text('model path', width=80), self.deploy_model_path,
                                                    ft.Text('', width=10)]),
                                        ])
                                    )
                                ]
                            ),
                            ft.ExpansionTile(
                                title=ft.Text("Input Settings"),
                                # subtitle=ft.Text("Deploy settings"),
                                initially_expanded=False,
                                controls=[
                                    ft.Card(
                                        ft.Column([
                                            ft.Row([ft.Text('CAM', width=80), self.deploy_input_CAM_type,
                                                    self.deploy_camera_dropdown, ft.Text('', width=10)]),
                                            ft.Row([ft.Text('imgsz', width=80), self.deploy_frame_width_input,
                                                    ft.Text('X'), self.deploy_frame_height_input,
                                                    ft.Text('', width=10)]),
                                            ft.Row([ft.Text('Confidence', width=80), self.deploy_settings_conf,
                                                    ft.Text('', width=10)]),
                                            ft.Row([ft.Text('IOU', width=80), self.deploy_settings_iou,
                                                    ft.Text('', width=10)]),
                                            ft.Row([ft.Text('Trigger type', width=80), self.deploy_input_trigger_type,
                                                    ft.Text('', width=10)]),
                                        ])
                                    )
                                ]
                            ),
                            ft.ExpansionTile(
                                title=ft.Text("Logic Settings"),
                                # subtitle=ft.Text("Deploy settings"),
                                initially_expanded=False,
                                controls=[
                                    ft.Card(
                                        ft.Column([
                                            ft.Row([
                                                ft.Text('Classes', width=80), self.deploy_logtic_class_select,
                                                ft.Text('Logic select'), self.deploy_logtic_type,
                                                ft.Text('', width=10),
                                            ],
                                                vertical_alignment=ft.CrossAxisAlignment.START),

                                        ])
                                    )
                                ]
                            ),
                            ft.ExpansionTile(
                                title=ft.Text("Output Settings"),
                                # subtitle=ft.Text("Deploy settings"),
                                initially_expanded=False,
                                controls=[
                                    ft.Card(
                                        ft.Column([
                                            ft.Row([ft.Text('Output type', width=80), self.deploy_output_type,
                                                    self.deploy_output_GPIO,
                                                    ft.Text('', width=10)]),

                                        ])
                                    )
                                ]
                            ),
                        ], scroll=ft.ScrollMode.ALWAYS)
                    )
                ]
            ),
        ], scroll=ft.ScrollMode.ALWAYS, spacing=0)

        self.t = ft.Tabs(selected_index=0, animation_duration=300, tabs=[
            ft.Tab(text="deploy", icon=ft.icons.FACT_CHECK, content=self.deploy_page),
            ft.Tab(text='Settings', icon=ft.icons.SETTINGS, content=self.settings_page)
        ], expand=1)

        self.setup_page()

    def remove_choice(self, e):
        # 删除选中的choice
        if self.deploy_choice_dropdown.value:
            try:
                with open(self.SETTINGS_FILE, 'r') as f:
                    settings = json.load(f)
                    settings.pop(self.deploy_choice_dropdown.value)
                with open(self.SETTINGS_FILE, 'w') as f:
                    json.dump(settings, f)
                self.deploy_choice_dropdown.options.clear()
                self.deploy_choice_dropdown.options = [ft.dropdown.Option(choice) for choice in settings.keys()]
                self.deploy_choice_dropdown.value = list(settings.keys())[0]
                self.deploy_choice_dropdown.update()
                self.load_settings()
                self.snack_message(f'Remove choice {self.deploy_choice_dropdown.value} success', 'green')
            except Exception as e:
                self.snack_message(f"Error remove choice: {e}", 'red')
        else:
            self.snack_message('Please select a choice', 'red')

    def add_choice(self, e):
        def on_dialog_submit(d):
            new_choice = d.content.value
            print(self.deploy_choice_dropdown.options)
            if new_choice:
                # check if choice already exists
                if new_choice in [option.key for option in self.deploy_choice_dropdown.options]:
                    self.snack_message(f'Choice {new_choice} already exists', 'red')
                    return
                self.deploy_choice_dropdown.options.append(ft.dropdown.Option(new_choice))
                self.deploy_choice_dropdown.value = new_choice
                self.deploy_choice_dropdown.update()
                self.save_settings()
            d.open = False
            d.update()
            self.snack_message(f'Add choice{new_choice} success', 'green')

        def on_dialog_cancel(d):
            d.open = False
            d.update()

        dialog = ft.AlertDialog(
            title=ft.Text("Add Choice"),
            content=ft.TextField(label="Enter choice"),
            actions=[
                ft.TextButton("Submit", on_click=lambda _: on_dialog_submit(dialog)),
                ft.TextButton("Cancel", on_click=lambda _: on_dialog_cancel(dialog))
            ]
        )
        self.page.dialog = dialog
        dialog.open = True
        self.page.update()

    def save_settings(self, e=None):
        try:
            settings = {}
            if os.path.exists(self.SETTINGS_FILE):
                with open(self.SETTINGS_FILE, 'r') as f:
                    settings = json.load(f)
            self.selected_classes = [control.label for control in self.deploy_logtic_class_select.controls if
                                     control.value]
            settings[self.deploy_choice_dropdown.value] = {
                'selected_project': self.selected_project,
                'model_path': self.model_path,
                'deploy_settings_history': self.deploy_settings_history.value,
                'deploy_settings_cam_type': self.deploy_input_CAM_type.value,
                'deploy_settings_camera': self.deploy_camera_dropdown.value,
                'deploy_settings_weight': self.deploy_settings_weight.value,
                'deploy_settings_conf': self.deploy_settings_conf.value,
                'deploy_settings_iou': self.deploy_settings_iou.value,
                'deploy_frame_width': self.deploy_frame_width_input.value,
                'deploy_frame_height': self.deploy_frame_height_input.value,
                'trigger_type': self.deploy_input_trigger_type.value,
                'all_classes': self.classes,
                'logtic_class_select': self.selected_classes,
                'logtic_type': self.deploy_logtic_type.value,
                'output_type': self.deploy_output_type.value,
                'output_GPIO': self.deploy_output_GPIO.value,
                'theme_mode': self.page.theme_mode.name
            }
            with open(self.SETTINGS_FILE, 'w') as f:
                json.dump(settings, f)
            self.snack_message('Save settings success', 'green')
        except Exception as e:
            self.snack_message(f"Error saving settings: {e}", 'red')

    def load_settings(self, e=None):
        try:
            with open(self.SETTINGS_FILE, 'r') as f:
                settings = json.load(f)
                if self.deploy_choice_dropdown.value is None:
                    self.deploy_choice_dropdown.options.clear()
                    self.deploy_choice_dropdown.options = [ft.dropdown.Option(choice) for choice in settings.keys()]
                    try:
                        self.deploy_choice_dropdown.value = list(settings.keys())[0]
                    except:
                        self.snack_message('No choice found', 'red')
                        return
                    self.deploy_choice_dropdown.update()
                else:
                    pass
                self.deploy_choice_text.value = self.deploy_choice_dropdown.value
                self.deploy_choice_text.update()
                settings = settings.get(self.deploy_choice_dropdown.value)
                self.selected_project = settings.get('selected_project')
                self.deploy_project_dropdown.value = self.selected_project
                self.model_path = settings.get('model_path')
                self.deploy_settings_history.options.clear()
                self.find_train_history(self.selected_project)
                self.deploy_settings_history.value = settings.get('deploy_settings_history')
                self.deploy_camera_dropdown.value = settings.get('deploy_settings_camera', '0')
                self.deploy_input_CAM_type.value = settings.get('deploy_settings_cam_type', 'CV CAM')

                self.deploy_settings_weight.options.clear()
                self.find_weights(None)
                self.deploy_settings_weight.value = settings.get('deploy_settings_weight')
                self.deploy_settings_conf.value = settings.get('deploy_settings_conf', 0.3)
                self.deploy_settings_iou.value = settings.get('deploy_settings_iou', 0.5)
                self.deploy_frame_width_input.value = settings.get('deploy_frame_width', '640')
                self.deploy_frame_height_input.value = settings.get('deploy_frame_height', '480')
                self.page.theme_mode = ft.ThemeMode[settings.get('theme_mode', 'LIGHT')]
                self.theme_switch.icon = ft.icons.WB_SUNNY_OUTLINED if self.page.theme_mode == ft.ThemeMode.LIGHT else ft.icons.NIGHTLIGHT_OUTLINED

                self.deploy_project_dropdown.update()
                self.deploy_settings_history.update()
                self.deploy_settings_weight.update()
                self.deploy_model_path.value = self.model_path
                self.deploy_model_path.update()
                self.theme_switch.update()
                self.deploy_input_CAM_type.update()

                self.classes = settings.get('all_classes', {})
                self.selected_classes = settings.get('logtic_class_select', [])
                self.deploy_logtic_class_select.controls = []
                for i, class_key in enumerate(self.classes.keys()):
                    self.deploy_logtic_class_select.controls.append(ft.Checkbox(
                        label=self.classes[class_key],
                        value=True if self.classes[class_key] in settings.get('logtic_class_select', []) else False))
                self.deploy_logtic_class_select.update()

                self.deploy_logtic_type.value = settings.get('logtic_type', [])
                self.deploy_logtic_type.update()

                self.deploy_input_trigger_type.value = settings.get('trigger_type', 'streaming')
                self.deploy_input_trigger_type.update()

                self.deploy_output_type.value = settings.get('output_type', 'Visualize')
                self.deploy_output_type.update()

                self.deploy_output_GPIO.value = settings.get('output_GPIO', 'GPIO_21')
                self.deploy_output_GPIO.update()

                self.page.update()
        except FileNotFoundError:
            self.snack_message('No settings file found', 'red')

    def change_theme(self, e):
        self.page.theme_mode = ft.ThemeMode.DARK if self.page.theme_mode == ft.ThemeMode.LIGHT else ft.ThemeMode.LIGHT
        self.theme_switch.icon = ft.icons.WB_SUNNY_OUTLINED if self.page.theme_mode == ft.ThemeMode.LIGHT else ft.icons.NIGHTLIGHT_OUTLINED
        self.theme_switch.update()
        self.page.update()
        self.save_settings()

    def setup_page(self):
        self.page.views.clear()
        self.page.views.append(ft.View("/", [ft.AppBar(leading=ft.Icon(ft.icons.CATCHING_POKEMON), adaptive=True,
                                                       title=ft.Text('Welcome to FAHAI deploy tool !'),
                                                       center_title=False, bgcolor=ft.colors.SURFACE_VARIANT,
                                                       actions=[self.theme_switch, ft.IconButton(ft.icons.EXIT_TO_APP,
                                                                                                 on_click=lambda
                                                                                                     e: self.page.window_close())]),
                                             self.t]))
        self.page.update()

    def find_classes(self, e):
        # find classes from .pt file
        from ultralytics import YOLO
        try:
            model = self.load_model(YOLO, self.model_path)
            self.classes = model.names
            print(self.classes)
            self.deploy_logtic_class_select.controls = []
            for i, class_key in enumerate(self.classes.keys()):
                self.deploy_logtic_class_select.controls.append(ft.Checkbox(
                    label=self.classes[class_key], value=False))

            self.deploy_logtic_class_select.update()
        except Exception as e:
            self.snack_message(f"Error finding weight path: {e}", 'red')

    def on_model_picked(self, e: ft.FilePickerResultEvent):
        if e.files:
            self.model_path = e.files[0].path
            self.deploy_model_path.value = self.model_path
            self.deploy_model_path.update()
            self.snack_message(f"Import model {self.model_path} success", 'green')
        self.find_classes(e)

    def find_weight_path(self, e):
        # find weight path
        self.model_path = os.path.join(os.getcwd(), 'projects', self.selected_project, 'train',
                                       self.deploy_settings_history.value, 'weights', self.deploy_settings_weight.value)
        self.deploy_model_path.value = self.model_path
        self.deploy_model_path.update()
        self.find_classes(e)

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
        self.camera_thread_instance = threading.Thread(target=self.camera_thread, args=(
            camera_index, self.deploy_img_element, predict_on, model_path))
        self.camera_thread_instance.do_run = True
        self.camera_thread_instance.start()

    def stop_deploy_camera(self, e=None):
        if self.camera_thread_instance:
            self.camera_thread_instance.do_run = False
            self.camera_thread_instance.join()
            self.snack_message('CAM is stopped', 'green')
            self.deploy_progress_bar.visible = False
            self.deploy_progress_bar.update()
            self.deploy_img_element.src_base64 = ""
            self.initialize_gpio()

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

    def initialize_gpio(self,e=None):
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            GPIO.setup(self.RelayA[0], GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(self.RelayA[1], GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(self.RelayA[2], GPIO.OUT, initial=GPIO.LOW)
            self.snack_message(
                f"GPIO initialized, RelayA: {self.RelayA[0]}, {self.RelayA[1]}, {self.RelayA[2]}", 'green'
            )
        except Exception as e:
            self.deploy_output_GPIO.value = "None"
            self.deploy_output_GPIO.update()
            self.snack_message(f"Error initializing GPIO: {e}", 'red')

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

        if self.deploy_input_CAM_type.value == "hikrobotic CAM":
            sys.path.append(os.path.join(os.getcwd(), 'hik_CAM'))
            try:
                from hik_CAM.getFrame import start_cam, exit_cam, get_frame
                self.cap, self.stOutFrame, self.data_buf = start_cam(nConnectionNum=camera_index)
            except Exception as e:
                self.snack_message(f"Error starting CAM: {e}", 'red')
                self.deploy_progress_bar.visible = True
                self.deploy_progress_bar.update()
                self.page.update()
                return
        elif self.deploy_input_CAM_type.value == "CV CAM":
            try:
                self.cap = cv2.VideoCapture(camera_index)
            except Exception as e:
                self.snack_message(f"Error starting CAM: {e}", 'red')
                self.deploy_progress_bar.visible = True
                self.deploy_progress_bar.update()
                self.page.update()
                return
        else:
            self.snack_message('select CAM type first ', 'red')
            self.stop_deploy_camera()

        while getattr(threading.currentThread(), "do_run", True):
            if self.deploy_input_CAM_type.value == "hikrobotic CAM":
                ret, frame = get_frame(self.cap, self.stOutFrame)
            elif self.deploy_input_CAM_type.value == "CV CAM":
                ret, frame = self.cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            conf = float(self.deploy_settings_conf.value)
            iou = float(self.deploy_settings_iou.value)
            width = int(self.deploy_frame_width_input.value)
            height = int(self.deploy_frame_height_input.value)
            try:
                res = model.predict(frame, conf=conf, iou=iou, imgsz=(width, height))

                res_plotted = res[0].plot()
                res_json = res[0].tojson()
                logic_result = self.logic_check(res_json)

                markdown_text = f"```dart\n{res_json}\n```"
                self.deploy_results.value = markdown_text
            except:
                res_plotted = frame
            if self.deploy_output_type.value == "Visualize":
                if logic_result :
                    res_plotted = cv2.putText(res_plotted, "Logic check pass", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                             (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    res_plotted = cv2.putText(res_plotted, "Logic check failed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                             (0, 0, 255), 2, cv2.LINE_AA)
            if self.deploy_output_GPIO.value == "GPIO_21":
                if logic_result:
                    GPIO.output(self.RelayA[0], GPIO.HIGH)
                else:
                    GPIO.output(self.RelayA[0], GPIO.LOW)

            img_pil = Image.fromarray(res_plotted)
            img_byte_arr = BytesIO()
            img_pil.save(img_byte_arr, format="JPEG")
            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            img_element.src_base64 = img_base64
            self.page.update()
            time.sleep(0.03)
        if self.deploy_input_CAM_type.value == "hikrobotic CAM":
            exit_cam(self.cap, self.data_buf)
        elif self.deploy_input_CAM_type.value == "CV CAM":
            self.cap.release()

    def logic_check(self, res_json) -> bool:
        res_json_load = json.loads(res_json)
        result_classes = [r['name'] for r in res_json_load]
        if self.deploy_logtic_type.value == "detected results [include] left selected classes":
            # if detected result_classes [include] left selected classes  -> true
            l='INCLODE'
            check_result=all(item in result_classes for item in self.selected_classes)

        elif self.deploy_logtic_type.value == "detected results [in] left selected classes":
            # if detected result_classes [in] left selected classes -> true
            l='IN'
            check_result=all(item in self.selected_classes for item in result_classes)

        elif self.deploy_logtic_type.value == "detected results & left selected classes [No intersection]":
            # if detected result_classes & left selected classes [No intersection] -> true
            l='NO INTERSECTION WITH'
            check_result=len(set(result_classes).intersection(set(self.selected_classes))) == 0
        elif self.deploy_logtic_type.value == "detected results & left selected classes [intersection]":
            # if detected result_classes & left selected classes [intersection] -> true
            l='INTERSECTION WITH'
            check_result=len(set(result_classes).intersection(set(self.selected_classes))) > 0
        elif self.deploy_logtic_type.value == "count":
            # count each kind of object in result_classes,and return a dict
            # return {item: result_classes.count(item) for item in set(result_classes)}
            l='COUNT'
            check_result = sum(result_classes.count(item) for item in set(self.selected_classes))

        print(f'detected:{result_classes} {l} target:{self.selected_classes} -> {check_result}')
        print('-------------------')
        return check_result


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
    # page.window_frameless = True
    # page.window_resizable = False
    page.bgcolor = ft.colors.BLUE_GREY_200
    page.window_maximized = True
    app = FAHAI(page)


if __name__ == "__main__":
    ft.app(target=main, assets_dir='assets')
