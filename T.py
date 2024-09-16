import subprocess

def run_label_studio(port=8080):
    # 启动 Label Studio，指定端口
    try:
        subprocess.run(['label-studio', 'start', '--port', str(port)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error starting Label Studio: {e}")
    except FileNotFoundError:
        print("Label Studio is not installed or not found in PATH.")

if __name__ == "__main__":
    # 指定端口运行 Label Studio
    run_label_studio(port=9000)  # 替换9000为你想要的端口号
