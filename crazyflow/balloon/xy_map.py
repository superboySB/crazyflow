"""Read-only, metric XY map geometry and native OpenGL drawing for frozen flights."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, log10
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Rect:
    """Framebuffer rectangle in bottom-left coordinates."""

    left: float
    bottom: float
    width: float
    height: float

    @property
    def right(self) -> float:
        """Exclusive right edge."""
        return self.left + self.width

    @property
    def top(self) -> float:
        """Exclusive top edge."""
        return self.bottom + self.height

    def contains(self, x: float, y: float) -> bool:
        """Route input only to the actual visible viewport."""
        return self.left <= x < self.right and self.bottom <= y < self.top

    def as_tuple(self) -> tuple:
        """Return integer coordinates suitable for MjrRect."""
        return tuple(int(value) for value in (self.left, self.bottom, self.width, self.height))


@dataclass(frozen=True)
class Layout:
    """Three viewports and their separate telemetry area."""

    world: Rect
    map: Rect
    fpv: Rect
    telemetry: Rect


def layout(width: int, height: int, aspect: float = 4 / 3) -> Layout:
    """Keep world above map and preserve the original right-hand FPV/telemetry area."""
    if width < 320 or height < 320 or aspect <= 0:
        raise ValueError("The replay layout needs a positive aspect and at least 320x320 pixels")
    left, available = int(width * 0.58), height - 148
    map_height = max(1, int(available * 0.43))
    fpv_height = min(int((width - left - 16) / aspect), height - 320)
    fpv_height = max(fpv_height, 1)
    fpv = Rect(left + 8, height - 74 - fpv_height, int(fpv_height * aspect), fpv_height)
    return Layout(
        world=Rect(0, 82 + map_height + 8, left, available - map_height - 8),
        map=Rect(0, 82, left, map_height),
        fpv=fpv,
        telemetry=Rect(left + 8, 82, width - left - 16, max(fpv.bottom - 92, 1)),
    )


@dataclass(frozen=True)
class Bounds:
    """Fixed world-metre extent, independent of playback time or window resizing."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def as_tuple(self) -> tuple:
        """Return x-min/max, y-min/max for the evidence sidecar."""
        return (self.xmin, self.xmax, self.ymin, self.ymax)


def episode_bounds(
    states: np.ndarray, initial_target: np.ndarray, velocity: np.ndarray, horizon: float
) -> Bounds:
    """Fit full finite aircraft records, the target route, and all 1 s arrow tips.

    Full records determine scale only. The renderer reveals just the aircraft's
    elapsed prefix; the prescribed target future is explicitly dashed SIM TRUTH.
    """
    states = np.asarray(states)
    if states.ndim != 2 or states.shape[-1] != 12:
        raise ValueError("Map bounds need at least one finite recorded 12D aircraft state")
    finite = states[np.isfinite(states).all(axis=-1)]
    if len(finite) == 0:
        raise ValueError("Map bounds need at least one finite recorded 12D aircraft state")
    target, velocity = np.asarray(initial_target), np.asarray(velocity)
    if target.shape != (3,) or velocity.shape != (3,) or not np.isfinite([target, velocity]).all():
        raise ValueError("Map target position and velocity must be finite 3D vectors")
    if not np.isfinite(horizon) or horizon < 0:
        raise ValueError("Map horizon must be finite and nonnegative")
    points = np.concatenate(
        (
            finite[:, :2],
            finite[:, :2] + finite[:, 6:8],
            target[None, :2] + np.array([0, horizon, horizon + 1])[:, None] * velocity[None, :2],
        )
    )
    low, high = points.min(axis=0), points.max(axis=0)
    center, span = (low + high) / 2, np.maximum(high - low, 1.0)
    padding = np.maximum(0.3, span * 0.08)
    low, high = center - span / 2 - padding, center + span / 2 + padding
    return Bounds(float(low[0]), float(high[0]), float(low[1]), float(high[1]))


@dataclass(frozen=True)
class Projection:
    """Letterbox a fixed metre rectangle with equal horizontal/vertical scale."""

    bounds: Bounds
    rect: Rect

    @property
    def pixels_per_m(self) -> float:
        """One shared scale for both world axes."""
        return min(
            self.rect.width / (self.bounds.xmax - self.bounds.xmin),
            self.rect.height / (self.bounds.ymax - self.bounds.ymin),
        )

    @property
    def plot_rect(self) -> Rect:
        """Actual plot rectangle after aspect-preserving letterboxing."""
        width = (self.bounds.xmax - self.bounds.xmin) * self.pixels_per_m
        height = (self.bounds.ymax - self.bounds.ymin) * self.pixels_per_m
        return Rect(
            self.rect.left + (self.rect.width - width) / 2,
            self.rect.bottom + (self.rect.height - height) / 2,
            width,
            height,
        )

    def project(self, xy: np.ndarray) -> np.ndarray:
        """Project +X right, +Y up without clamping or distorting line segments."""
        return (np.asarray(xy) - [self.bounds.xmin, self.bounds.ymin]) * self.pixels_per_m + [
            self.plot_rect.left,
            self.plot_rect.bottom,
        ]


def motion_kind(velocity: np.ndarray) -> str:
    """Distinguish stationary and vertical motion without inventing XY direction."""
    velocity = np.asarray(velocity)
    if np.linalg.norm(velocity) <= 1e-8:
        return "stationary"
    return "vertical" if np.linalg.norm(velocity[:2]) <= 1e-8 else "moving"


def velocity_arrow(position: np.ndarray, velocity: np.ndarray, seconds: float = 1.0) -> np.ndarray:
    """A shaft/two-wing polyline: start, tip, wing, tip, wing; metres, not unit length."""
    delta = np.asarray(velocity)[:2] * seconds
    length = np.linalg.norm(delta)
    if length <= 1e-8:
        return np.empty((0, 2))
    start = np.asarray(position)[:2]
    tip, direction = start + delta, delta / length
    normal = np.array([-direction[1], direction[0]])
    wing = min(length * 0.25, 0.18)
    return np.stack(
        (
            start,
            tip,
            tip - direction * wing + normal * wing / 2,
            tip,
            tip - direction * wing - normal * wing / 2,
        )
    )


def trajectory_prefix(states: np.ndarray, sample_dt: float, elapsed: float) -> np.ndarray:
    """Reveal only past positions and the current interpolated endpoint, never a future state."""
    states = np.asarray(states)
    when = float(np.clip(elapsed, 0, (len(states) - 1) * sample_dt))
    index = min(int(when / sample_dt), len(states) - 1)
    fraction = when / sample_dt - index
    current = (
        states[index, :3] * (1 - fraction) + states[min(index + 1, len(states) - 1), :3] * fraction
    )
    points = np.concatenate((states[: index + 1, :3], current[None]))
    return points[np.isfinite(points).all(axis=-1)]


def _grid_step(pixels_per_m: float) -> float:
    raw = 55 / pixels_per_m
    base = 10 ** floor(log10(raw))
    return next(multiplier * base for multiplier in (1, 2, 5, 10) if multiplier * base >= raw)


def draw_map(
    replay: Any, rect: Rect, bounds: Bounds, context: Any, framebuffer: tuple[int, int]
) -> dict:
    """Draw recorded truth only, using OpenGL already required by the MuJoCo image.

    Scissoring clips complete segments rather than clamping their endpoints to
    the map border. Every OpenGL state/matrix change is restored before returning.
    """
    import mujoco
    from OpenGL import GL

    def text(area: Rect, value: str) -> None:
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            mujoco.MjrRect(*area.as_tuple()),
            value,
            "",
            context,
        )

    projection = Projection(
        bounds,
        Rect(rect.left + 52, rect.bottom + 32, max(rect.width - 72, 1), max(rect.height - 136, 1)),
    )
    plot, step = projection.plot_rect, _grid_step(projection.pixels_per_m)
    state, target, velocity = replay.state(), replay.target, replay.target_velocity
    elapsed = min(replay.elapsed, replay.duration)
    records = replay.current["substep_states"][:, replay.episode]
    horizon = (len(records) - 1) * replay.cfg.dt / replay.cfg.substeps
    past = trajectory_prefix(records, replay.cfg.dt / replay.cfg.substeps, elapsed)
    if len(past) > 300:
        past = past[np.unique(np.r_[np.linspace(0, len(past) - 1, 300).astype(int), len(past) - 1])]
    cyan, orange = (0.15, 0.88, 1.0), (1.0, 0.36, 0.12)
    mujoco.mjr_rectangle(mujoco.MjrRect(*rect.as_tuple()), 0.055, 0.08, 0.105, 1)
    GL.glPushAttrib(GL.GL_ALL_ATTRIB_BITS)
    old_matrix = int(GL.glGetIntegerv(GL.GL_MATRIX_MODE))
    GL.glMatrixMode(GL.GL_PROJECTION)
    GL.glPushMatrix()
    GL.glLoadIdentity()
    GL.glOrtho(0, framebuffer[0], 0, framebuffer[1], -1, 1)
    GL.glMatrixMode(GL.GL_MODELVIEW)
    GL.glPushMatrix()
    GL.glLoadIdentity()
    try:
        GL.glViewport(0, 0, *framebuffer)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_LIGHTING)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glEnable(GL.GL_SCISSOR_TEST)
        GL.glScissor(*plot.as_tuple())

        def line(points: np.ndarray, color: tuple, width: float = 1) -> None:
            if len(points) < 2:
                return
            GL.glColor3f(*color)
            GL.glLineWidth(width)
            GL.glBegin(GL.GL_LINE_STRIP)
            for x, y in projection.project(np.asarray(points)[..., :2]):
                GL.glVertex2f(float(x), float(y))
            GL.glEnd()

        xs = np.arange(ceil(bounds.xmin / step), floor(bounds.xmax / step) + 1) * step
        ys = np.arange(ceil(bounds.ymin / step), floor(bounds.ymax / step) + 1) * step
        for x in xs:
            line(np.array([[x, bounds.ymin], [x, bounds.ymax]]), (0.2, 0.28, 0.35))
        for y in ys:
            line(np.array([[bounds.xmin, y], [bounds.xmax, y]]), (0.2, 0.28, 0.35))
        line(past, cyan, 2)
        line(np.stack((replay.initial_target, target)), orange, 2)
        # Each dash is a proper world-space segment, not a repeated/reframed path.
        future = replay.target_at(horizon)
        future_pixels = np.linalg.norm(
            projection.project(future[:2]) - projection.project(target[:2])
        )
        dashes = max(1, int(ceil(future_pixels / 12)))
        for index in range(dashes):
            fractions = np.array([index, index + 0.5]) / dashes
            line(target[None] + fractions[:, None] * (future - target)[None], (0.65, 0.3, 0.17))
        line(velocity_arrow(state[:3], state[6:9]), cyan, 2)
        line(velocity_arrow(target, velocity), orange, 2)
        for position, color, size in ((state[:2], cyan, 8), (target[:2], orange, 10)):
            GL.glColor3f(*color)
            GL.glPointSize(size)
            GL.glBegin(GL.GL_POINTS)
            GL.glVertex2f(*projection.project(position))
            GL.glEnd()
    finally:
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPopMatrix()
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glPopMatrix()
        GL.glMatrixMode(old_matrix)
        GL.glPopAttrib()

    for x in xs:
        pixel = projection.project([x, bounds.ymin])
        text(Rect(pixel[0] - 12, plot.bottom - 25, 52, 22), f"{0 if abs(x) < 1e-9 else x:g}")
    for y in ys:
        pixel = projection.project([bounds.xmin, y])
        text(Rect(plot.left - 45, pixel[1] - 9, 42, 22), f"{0 if abs(y) < 1e-9 else y:g}")
    text(Rect(rect.right - 64, rect.bottom + 3, 64, 22), "+X [m]")
    text(Rect(rect.left + 4, plot.top - 20, 60, 22), "+Y [m]")
    xyz = "/".join(f"{value:+.2f}" for value in velocity)
    header = (
        "XY MAP | SIM TRUTH, not actor inputs | +X right / +Y up\n"
        f"Cyan drone; orange target; dashed future; arrows 1s; grid {step:g}m\n"
        f"Target vXYZ {xyz}m/s | 3D {np.linalg.norm(velocity):.2f}m/s {motion_kind(velocity)}\n"
        f"z[m]/vz[m/s]: drone {state[2]:.2f}/{state[8]:+.2f} | "
        f"target {target[2]:.2f}/{velocity[2]:+.2f} | dZ {target[2] - state[2]:+.2f}m"
    )
    text(Rect(rect.left + 6, rect.top - 98, rect.width - 12, 98), header)
    return {
        "bounds_m": bounds.as_tuple(),
        "coordinate_axes": "+X right; +Y up; world metres, fixed episode bounds and equal scale",
        "velocity_arrow_seconds": 1.0,
        "actor_excluded": True,
        "motion": motion_kind(velocity),
        "aircraft_position_m": state[:3].tolist(),
        "aircraft_velocity_m_s": state[6:9].tolist(),
        "target_position_m": target.tolist(),
        "target_velocity_m_s": velocity.tolist(),
        "time_s": elapsed,
        "pixels_per_m": projection.pixels_per_m,
        "role": "SIM TRUTH, excluded from actor; dashed target future is prescribed, not measured",
    }
