class Output:
    _show = True
    _warn = True

    @classmethod
    def set_show_messages(cls, show: bool) -> None:
        cls._show = show

    @classmethod
    def set_show_warnings(cls, show: bool) -> None:
        cls._warn = show
