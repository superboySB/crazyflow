"""Locate editable outer resources in a checkout or a self-contained wheel install."""

from __future__ import annotations

import os
import sysconfig
from pathlib import Path


def resource_root() -> Path:
    """Return the checkout root or installed ``share/crazyflow`` resource directory.

    An explicit CRAZYFLOW_RESOURCE_ROOT names a complete resource tree containing
    ``assets/`` and ``config/``. It never silently falls back to a different tree.
    Normal source and Docker runs need no environment variable or host resources.
    """
    explicit = os.environ.get("CRAZYFLOW_RESOURCE_ROOT")
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if (root / "assets").is_dir() and (root / "config").is_dir():
            return root
        raise FileNotFoundError(
            f"CRAZYFLOW_RESOURCE_ROOT must contain assets/ and config/: {root}"
        )
    module = Path(__file__).resolve()
    checkout = module.parents[1]
    if (
        (checkout / "pyproject.toml").is_file()
        and (checkout / "assets").is_dir()
        and (checkout / "config").is_dir()
    ):
        return checkout
    # Also cover pip --user, --prefix and --target installs whose data root may
    # differ from sysconfig. Prefer the imported package's own installation tree
    # over another global installation with different resource/config versions.
    candidates = [parent / "share" / "crazyflow" for parent in module.parents]
    candidates.append(Path(sysconfig.get_path("data")) / "share" / "crazyflow")
    for candidate in candidates:
        if (candidate / "assets").is_dir() and (candidate / "config").is_dir():
            return candidate
    raise FileNotFoundError(
        "Crazyflow assets/config are missing. Install the complete wheel or use a full checkout."
    )


def resource_path(relative: str | Path) -> Path:
    """Return an existing resource under the selected root, independent of cwd."""
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Resource paths must be relative to the Crazyflow resource root")
    path = resource_root() / relative
    if not path.exists():
        raise FileNotFoundError(f"Crazyflow resource is missing: {path}")
    return path
