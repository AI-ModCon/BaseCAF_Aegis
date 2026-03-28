"""Aegis: HPC LLM Instance Launcher."""

__version__ = "0.1.0"

import pathlib as _pathlib
import warnings as _warnings

_CANONICAL_ROOT = _pathlib.Path(__file__).resolve().parent

# Guard against stale copies in site-packages (e.g. from `module load frameworks`)
# shadowing the editable development install.
if "site-packages" in str(_CANONICAL_ROOT) and not str(_CANONICAL_ROOT).startswith(
    str(_pathlib.Path.home() / "Aegis")
):
    _warnings.warn(
        f"aegis is being imported from a non-development path: {_CANONICAL_ROOT}\n"
        "This may be a stale copy. Ensure the editable install at ~/Aegis is first "
        "in sys.path, or remove the stale copy.",
        ImportWarning,
        stacklevel=2,
    )
