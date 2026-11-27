from __future__ import annotations

from pathlib import Path


def _resolve_asset(relative_path: str) -> str:
    """Resolve only the assets bundled beside this evaluation package."""
    candidate = Path(__file__).resolve().parents[2] / "assets" / relative_path
    if not candidate.is_file():
        raise FileNotFoundError(f"Missing bundled Isaac asset: {candidate}")
    return str(candidate)


def get_crazyflie_usd_path() -> str:
    return _resolve_asset("Assets/Isaac/4.5/Isaac/Robots/Crazyflie/cf2x.usd")


def get_ui_arrow_usd_path() -> str:
    return _resolve_asset("Assets/Isaac/4.5/Isaac/Props/UIElements/arrow_x.usd")


def get_ui_frame_usd_path() -> str:
    return _resolve_asset("Assets/Isaac/4.5/Isaac/Props/UIElements/frame_prim.usd")
