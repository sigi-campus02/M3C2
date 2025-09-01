from importlib import reload
from unittest.mock import patch


def test_main_invokes_cliapp_run_once():
    """Ensure that main.main runs CLIApp.run exactly once and is import-safe."""
    with patch("m3c2.cli.cli.CLIApp.run") as mock_run:
        import main  # Import after patching to ensure no side effects
        main = reload(main)

        mock_run.assert_not_called()
        main.main()
        mock_run.assert_called_once()
