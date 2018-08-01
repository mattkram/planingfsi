from planingfsi.command_line.planingfsi import planingfsi


from click.testing import CliRunner


def test_run_main_cli():
    runner = CliRunner()

    results = runner.invoke(planingfsi)

    print(results)
