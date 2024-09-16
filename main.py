import flet as ft
from fahai import FAHAI

def main(page: ft.Page):
    page.title = "BOSCH_HzP_AI"
    page.padding = 0
    page.theme = ft.theme.Theme(font_family="Verdana")
    page.theme.page_transitions.windows = "cupertino"
    page.fonts = {"Pacifico": "Pacifico-Regular.ttf"}
    page.bgcolor = ft.colors.BLUE_GREY_200
    page.window_maximized = True

    app = FAHAI(page)

ft.app(target=main)