"""Functions for writing results to a file on disk."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import TextIO


def write_as_dict(filename: Path | str, *args: Any, data_format: str = ">10.8e") -> None:
    """Write arguments to a file as a dictionary."""
    with Path(filename).open("w") as ff:
        for name, value in args:
            ff.write(f"{name:<14} : {value:{data_format}}\n")


def write_as_list(
    filename: Path | str, *args: Any, header_format: str = "<15", data_format: str = ">+10.8e"
) -> None:
    """Write the arguments to a file as a list."""
    with Path(filename).open("w") as ff:
        _write(ff, header_format, [item for item in [arg[0] for arg in args]])
        for value in zip(*[arg[1] for arg in args]):
            _write(ff, data_format, value)


def _write(ff: TextIO, write_format: str, items: Any) -> None:
    """Write items to a file with a specific format."""
    if isinstance(items[0], str):
        ff.write("# ")
    else:
        ff.write("  ")
    ff.write("".join("{1:{0}} ".format(write_format, item) for item in items) + "\n")
