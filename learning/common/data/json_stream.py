"""Incrementally read a top-level JSON array without materializing the file."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import re
from typing import Iterator, TextIO


_CHUNK_SIZE = 1 << 20


def _array_pattern(name: str) -> re.Pattern[str]:
    return re.compile(rf'"{re.escape(name)}"\s*:\s*\[')


def _read_array_start(stream: TextIO, name: str) -> tuple[dict, str]:
    pattern = _array_pattern(name)
    text = ""
    while True:
        chunk = stream.read(_CHUNK_SIZE)
        text += chunk
        match = pattern.search(text)
        if match is not None:
            header_text = text[: match.start()] + json.dumps(name) + ": []}"
            header = json.loads(header_text)
            if not isinstance(header, dict):
                raise ValueError("The JSON root must be an object")
            return header, text[match.end() :]
        if not chunk:
            raise ValueError(f"Missing top-level {name!r} array")
        # Dataset headers are deliberately small. A very large prefix generally
        # means the requested key is absent or nested in an unsupported envelope.
        if len(text) > 64 * _CHUNK_SIZE:
            raise ValueError(f"The {name!r} array was not found in the JSON header")


def _iter_array_values(stream: TextIO, initial: str) -> Iterator[dict]:
    decoder = json.JSONDecoder()
    buffer = initial
    position = 0
    eof = False
    while True:
        while position < len(buffer) and buffer[position] in " \t\r\n,":
            position += 1
        if position < len(buffer) and buffer[position] == "]":
            return
        try:
            value, end = decoder.raw_decode(buffer, position)
        except json.JSONDecodeError as error:
            if eof:
                raise ValueError("Truncated or invalid JSON array") from error
            buffer = buffer[position:]
            position = 0
            chunk = stream.read(_CHUNK_SIZE)
            eof = not chunk
            buffer += chunk
            continue
        if not isinstance(value, dict):
            raise ValueError("Dataset array entries must be JSON objects")
        yield value
        position = end
        if position > _CHUNK_SIZE:
            buffer = buffer[position:]
            position = 0


@contextmanager
def open_object_array(
    path: str | Path, array_name: str
) -> Iterator[tuple[dict, Iterator[dict]]]:
    """Yield an object header and a streaming iterator over its final array.

    Published AI Factory datasets place their large row array last in the root
    object. Keeping this restriction explicit prevents a permissive reader from
    silently ignoring metadata written after that array.
    """

    with Path(path).open(encoding="utf-8") as stream:
        header, remainder = _read_array_start(stream, array_name)
        yield header, _iter_array_values(stream, remainder)
