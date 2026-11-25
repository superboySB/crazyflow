"""Hardware descriptions for the supported drone platforms.

Outer ``assets/drones/`` contains each MuJoCo MJCF and its referenced meshes.
For physical params, see [crazyflow.dynamics.load_params][].

Use ``Drone`` to enumerate the supported configurations.
"""

from enum import StrEnum
from pathlib import Path

from crazyflow.paths import resource_path

__all__ = ["Drone"]


class Drone(StrEnum):
    """Drone configurations with an outer ``assets/drones/<name>.xml`` resource."""

    cf21B_500 = "cf21B_500"
    cf2x_L250 = "cf2x_L250"
    cf2x_P250 = "cf2x_P250"
    cf2x_T350 = "cf2x_T350"
    hb_x500 = "hb_x500"

    @property
    def path(self) -> Path:
        """Return the MJCF path in a source checkout or installed resource tree."""
        return resource_path(f"assets/drones/{self.value}.xml")


# Sanity check at startup
_mjcf_files = {p.stem for p in resource_path("assets/drones").glob("*.xml")}
assert {d.value for d in Drone} == _mjcf_files, (
    f"Drone enum {sorted(d.value for d in Drone)} does not match MJCF files {sorted(_mjcf_files)}"
)
