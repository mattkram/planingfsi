from __future__ import annotations

import json
import os
import re
from pathlib import Path
from re import Match
from typing import Any

from planingfsi import logger
from planingfsi import unit  # noqa: F401

__all__ = ["load_dict_from_file"]


def replace_single_quotes_with_double_quotes(string: str) -> str:
    """Replace all single-quoted strings with double-quotes."""

    def repl(m: Match) -> str:
        return m.group(1).join('""')

    return re.sub(r"'(.+?)'", repl, string)


def replace_environment_variables(string: str) -> str:
    """Replace environment variables with their value."""

    def repl(m: Match) -> str:
        return os.environ[m.group(1)]

    return re.sub(r"\$(\w+)", repl, string)


def add_quotes_to_words(string: str) -> str:
    """Find words inside a string and surround with double-quotes."""
    quoted_pattern = re.compile('(".+?")')
    word_pattern = re.compile(r"([\w.-]+)")
    # Any number, integer, float, or exponential
    number_pattern = re.compile(r"[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?")

    matches = quoted_pattern.split(string)

    def repl(m: Match) -> str:
        """Add quotes, only if it isn't a number."""
        value = m.group(1)
        if number_pattern.match(value):
            return value
        return value.join('""')

    return "".join(word_pattern.sub(repl, m) if i % 2 == 0 else m for i, m in enumerate(matches))


def jsonify_string(string: str) -> str:
    """Loop through a string, ensuring double-quotes are used to comply with json standard.

    * Find pattern.
    * Add everything up until the pattern to new copy of string.
    * Add pattern with single-quotes substituted for double-quotes.
    * If there's a double-quote inside single-quotes, it means we have an apostrophe.
    * Add everything up to the double quote.
    * Once they are all added, which will eventually include the apostrophe, normal matching will
      proceed.

    """
    if not string.startswith("{"):
        string = string.join("{}")

    string = replace_single_quotes_with_double_quotes(string)
    string = add_quotes_to_words(string)
    string = replace_environment_variables(string)

    string = re.sub(",+", ",", string)
    string = (
        string.replace("[,", "[")
        .replace("{,", "{")
        .replace(",]", "]")
        .replace(",}", "}")
        .replace("}{", "},{")
    )

    logger.debug(f'JSONified string: "{string}"')

    return string


def load_dict_from_file(
    filename: Path | str, key_map: dict[str, str] | None = None
) -> dict[str, Any]:
    """Read a file, which is a less strict JSON format, and return a dictionary.

    Optionally, a key map may be provided to allow loading older files by
    replacing keys with updated spellings. For example, a `key_map = {"oldKey": "old_key"}`
    could read a file containing the key "oldKey", but the dictionary that is
    returned will have replaced that key with "old_key". When using the `key_map`
    functionality, an exception will be raised if both the old and new keys exist
    in the dictionary being loaded.

    Args:
        filename: A filename or path to the file to be loaded.
        key_map: An optional mapping of keys.

    Returns:
        A dictionary mapping keys to values from the input file.

    """
    logger.debug('Loading Dictionary from file "{}"'.format(filename))

    with Path(filename).open() as f:
        dict_iter = (line.split("#")[0].strip() for line in f.readlines())
    try:
        dict_ = load_dict_from_string(",".join(dict_iter))
    except ValueError:
        print(f"Error reading file {filename}")
        raise

    # If specified, read values from a base dictionary
    # All local values override the base dictionary values
    base_dict_dir = dict_.get("baseDict", dict_.get("base_dict"))
    if base_dict_dir:
        base_dict_dir = os.path.split(base_dict_dir)
        # Tracing relative references from original file directory
        if base_dict_dir[0].startswith("."):
            base_dict_dir = os.path.abspath(os.path.join(os.path.dirname(filename), *base_dict_dir))
        base_dict = load_dict_from_file(base_dict_dir)
        dict_.update({k: v for k, v in base_dict.items() if k not in dict_})

    if key_map:
        dict_ = _apply_key_map(dict_, key_map)

    return dict_


def _apply_key_map(dict_: dict[str, Any], key_map: dict[str, str]) -> dict[str, Any]:
    """Map old keys to new keys."""
    for old_key, new_key in key_map.items():
        if old_key in dict_ and new_key in dict_:
            raise KeyError(f"Cannot use both '{old_key}' and '{new_key}'")
        if old_key in dict_:
            dict_[new_key] = dict_.pop(old_key)
    return dict_


def load_dict_from_string(string: str) -> dict[str, Any]:
    """Convert string to JSON string, convert to a dictionary, and return."""
    logger.debug('Loading Dictionary from string: "{}"'.format(string))

    json_string = jsonify_string(string)
    try:
        dict_ = json.loads(json_string)
    except json.decoder.JSONDecodeError:
        raise ValueError('Error loading from json string: "{}"'.format(json_string))

    # Provide specialized handling of certain strings
    for key, val in dict_.items():
        if isinstance(val, str):
            match = re.match(r"([+-]?nan|[+-]?inf)", val)
            if match:
                dict_[key] = float(match.group(1))
            elif "unit." in val:
                dict_[key] = eval(val)

    return dict_
