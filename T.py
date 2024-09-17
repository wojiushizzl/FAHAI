import flet as ft

def main(page: ft.Page):
    page.add(
        ft.SelectionArea(
            content=ft.Column([ft.Text("可选择的文本"), ft.Text("也是可选择的")])
        )
    )
    page.add(ft.Text("不可选择"))

ft.app(main)