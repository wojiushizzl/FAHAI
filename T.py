import flet as ft

def main(page):
    page.title = "ListTile 示例"
    page.add(
        ft.Card(
            content=ft.Container(
                width=500,
                content=ft.Column(
                    [
                        ft.ListTile(
                            title=ft.Text("一行列表Tile"),
                        ),
                        ft.ListTile(title=ft.Text("一行紧凑列表Tile"), dense=True),
                        ft.ListTile(
                            leading=ft.Icon(ft.icons.SETTINGS),
                            title=ft.Text("一行选中列表Tile"),
                            selected=True,
                        ),
                        ft.ListTile(
                            leading=ft.Image(src="/icons/icon-192.png", fit="contain"),
                            title=ft.Text("一行带leading控件"),
                        ),
                        ft.ListTile(
                            title=ft.Text("一行带trailing控件"),
                            trailing=ft.PopupMenuButton(
                                icon=ft.icons.MORE_VERT,
                                items=[
                                    ft.PopupMenuItem(text="Item 1"),
                                    ft.PopupMenuItem(text="Item 2"),
                                ],
                            ),
                        ),
                        ft.ListTile(
                            leading=ft.Icon(ft.icons.ALBUM),
                            title=ft.Text("一行带leading和trailing控件"),
                            trailing=ft.PopupMenuButton(
                                icon=ft.icons.MORE_VERT,
                                items=[
                                    ft.PopupMenuItem(text="Item 1"),
                                    ft.PopupMenuItem(text="Item 2"),
                                ],
                            ),
                        ),
                        ft.ListTile(
                            leading=ft.Icon(ft.icons.SNOOZE),
                            title=ft.Text("两行带leading和trailing控件"),
                            subtitle=ft.Text("这是第二个标题。"),
                            trailing=ft.PopupMenuButton(
                                icon=ft.icons.MORE_VERT,
                                items=[
                                    ft.PopupMenuItem(text="Item 1"),
                                    ft.PopupMenuItem(text="Item 2"),
                                ],
                            ),
                        ),
                    ],
                    spacing=0,
                ),
                padding=ft.padding.symmetric(vertical=10),
            )
        )
    )

ft.app(target=main)