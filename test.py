from piou import Cli, Option

cli = Cli(description='A CLI tool')




sub_cmd = cli.add_sub_parser(cmd='sub', help='A sub command')
sub_cmd.add_option('--test', help='Test mode')


@sub_cmd.command(cmd='bar', help='Run bar command')
def sub_bar_main(**kwargs):
    pass


@cli.command(cmd='foo', help='Run foo command')
def foo_main(
        bar: int = Option(..., help='Bar positional argument (required)'),
        baz: str = Option(..., '-b', '--baz', help='Baz keyword argument (required)'),
        foo: str = Option(None, '--foo', help='Foo keyword argument'),
):
    """
    A longer description on what the function is doing.  
    You can run it with:
    ```bash
     poetry run python -m piou.test.simple foo 1 -b baz
    ```
    And you are good to go!
    """
    pass


if __name__ == '__main__':
    cli.run()