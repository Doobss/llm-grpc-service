import click


class _Echo:

    _verbose_level: int = 0

    @classmethod
    def set_verbose_level(cls, level: int) -> None:
        cls._verbose_level = level

    @classmethod
    def _echo_when_verbose_level_ge(cls, level: int, text: str, **kwargs) -> None:
        if cls._verbose_level >= level:
            click.echo(message=text, **kwargs)

    def __call__(self, text: str, **kwargs) -> None:
        self._echo_when_verbose_level_ge(0, text, **kwargs)

    @classmethod
    def always(cls, text: str, **kwargs) -> None:
        cls._echo_when_verbose_level_ge(0, text, **kwargs)

    @classmethod
    def verbose(cls, text: str, count: int = 1, **kwargs) -> None:
        cls._echo_when_verbose_level_ge(count, text, **kwargs)


echo = _Echo()
_grey_rgb = (111, 113, 115)


def with_color(text: str, color) -> str:
    return click.style(text, fg=color)


def grey(text: str) -> str:
    return with_color(text, _grey_rgb)


def green(text: str) -> str:
    return with_color(text, 'green')


def red(text: str) -> str:
    return with_color(text, 'red')


def yellow(text: str) -> str:
    return with_color(text, 'yellow')


def cyan(text: str) -> str:
    return with_color(text, 'cyan')


def magenta(text: str) -> str:
    return with_color(text, 'magenta')