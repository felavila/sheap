"""Run `python -m sheap`.

Allow running SHEAP, also by invoking
the python module:

`python -m sheap`

This is an alternative to directly invoking the cli that uses python as the
"entrypoint".
"""

from __future__ import absolute_import

from sheap.cli import main

if __name__ == "__main__":  # pragma: no cover
    main(prog_name="sheap")  # pylint: disable=unexpected-keyword-arg
