import flet as ft
import Jetson.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
# GPIO.setwarnings(False)

RelayA = [21, 20, 26]

GPIO.setup(RelayA[0], GPIO.OUT, initial=GPIO.HIGH)
GPIO.setup(RelayA[1], GPIO.OUT, initial=GPIO.HIGH)
GPIO.setup(RelayA[2], GPIO.OUT, initial=GPIO.HIGH)


def main(page):
    def button_clicked(e):
        t.value = f"你选择的颜色是：{cg.value}"
        page.update()
        if cg.value=='red':
            print(cg.value)
            GPIO.output(RelayA[0], GPIO.HIGH)
            GPIO.output(RelayA[1], GPIO.LOW)
            GPIO.output(RelayA[2], GPIO.LOW)
        elif cg.value=='green':
            print(cg.value)

            GPIO.output(RelayA[0], GPIO.LOW)
            GPIO.output(RelayA[1], GPIO.HIGH)
            GPIO.output(RelayA[2], GPIO.LOW)
        elif cg.value=='blue':
            print(cg.value)

            GPIO.output(RelayA[0], GPIO.LOW)
            GPIO.output(RelayA[1], GPIO.LOW)
            GPIO.output(RelayA[2], GPIO.HIGH)



    t = ft.Text()
    b = ft.ElevatedButton(text="提交", on_click=button_clicked)
    cg = ft.RadioGroup(
        content=ft.Column(
            [
                ft.CupertinoRadio(value="red", label="红色 - Cupertino 单选按钮", active_color=ft.colors.RED, inactive_color=ft.colors.RED),
                ft.Radio(value="green", label="绿色 - Material 单选按钮", fill_color=ft.colors.GREEN),
                ft.Radio(value="blue", label="蓝色 - 自适应单选按钮", adaptive=True, active_color=ft.colors.BLUE),
            ]
        )
    )

    page.add(ft.Text("请选择你喜欢的颜色:"), cg, b, t)


ft.app(target=main)