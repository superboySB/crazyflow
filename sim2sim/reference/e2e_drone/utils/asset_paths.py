from __future__ import annotations

import os
from pathlib import Path


def _asset_roots() -> list[Path]:
    candidates: list[Path] = []
    asset_root = os.environ.get("CRAZYRL_ASSET_ROOT")
    if asset_root:
        candidates.append(Path(asset_root))
    candidates.append(Path("/workspace/CrazyRL/assets"))
    candidates.append(Path(__file__).resolve().parents[2] / "assets")
    return candidates


def _resolve_asset(relative_path: str) -> str:
    for root in _asset_roots():
        candidate = root / relative_path
        if candidate.exists():
            return str(candidate)
    return str(_asset_roots()[0] / relative_path)


def get_crazyflie_usd_path() -> str:
    return _resolve_asset("Assets/Isaac/4.5/Isaac/Robots/Crazyflie/cf2x.usd")


def get_ui_arrow_usd_path() -> str:
    return _resolve_asset("Assets/Isaac/4.5/Isaac/Props/UIElements/arrow_x.usd")


def get_ui_frame_usd_path() -> str:
    return _resolve_asset("Assets/Isaac/4.5/Isaac/Props/UIElements/frame_prim.usd")
