import flet as ft

def main(page: ft.Page):
    def find_option(option_name):
        for option in d.options:
            if option_name == option.key:
                return option
        return None

    def add_clicked(e):
        d.options.append(ft.dropdown.Option(option_textbox.value))
        d.value = option_textbox.value
        option_textbox.value = ""
        page.update()

    def delete_clicked(e):
        option = find_option(d.value)
        if option!= None:
            d.options.remove(option)
            # d.value = None
            page.update()

    d = ft.Dropdown()
    option_textbox = ft.TextField(hint_text="输入项目名称")
    add = ft.ElevatedButton("添加", on_click=add_clicked)
    delete = ft.OutlinedButton("删除所选项目", on_click=delete_clicked)
    #如何同时更新两个dropdown的值

    page.add(d,d, ft.Row(controls=[option_textbox, add, delete]))

ft.app(target=main)