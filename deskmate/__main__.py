"""Enable ``python -m deskmate`` → the same Typer app as the ``deskmate`` CLI.

This makes the package runnable without relying on the console-script shim
(``Scripts/deskmate.exe``), which matters for the GPU-training launcher that
invokes ``<env>/python.exe -m deskmate ui`` from a specific environment.
"""

from deskmate.engine.cli import app

if __name__ == "__main__":
    app()
