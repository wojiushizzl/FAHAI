import flet as ft
import subprocess
import threading

def main(page: ft.Page):
    # 创建一个Text组件用于显示控制台输出
    console_output = ft.Text("控制台输出：", width=800, height=400, color="black")

    # 用于启动子进程并实时更新控制台输出的函数
    def run_command(command):
        try:
            # 启动子进程
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)

            # 实时读取子进程的输出
            while True:
                output = process.stdout.readline()
                if output == "" and process.poll() is not None:
                    break
                if output:
                    # 更新控制台输出的文本内容
                    console_output.value += output
                    page.update()

            # 捕获并显示错误输出
            error_output = process.stderr.read()
            if error_output:
                console_output.value += f"\n错误信息：{error_output}"
                page.update()

        except Exception as e:
            console_output.value += f"\n执行出错：{str(e)}"
            page.update()

    # 启动子进程的线程
    def start_process(e):
        print(123123)
        command = "ping localhost"  # 这里可以替换为你想运行的命令
        threading.Thread(target=run_command, args=(command,)).start()

    # 创建一个按钮用于启动控制台命令
    run_button = ft.ElevatedButton(text="运行命令", on_click=start_process)

    # 将按钮和控制台输出添加到页面
    page.add(run_button, console_output)

# 运行Flet应用
ft.app(target=main)
