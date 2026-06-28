"""Visualizer color palette and node styling constants."""

from __future__ import annotations

GREY = (120, 120, 120)

DARK_BLUE = (20, 70, 175)
LIGHT_BLUE = (185, 215, 245)

OBSERVED_COLOR = (60, 130, 240)
SURVEILLED_COLOR = (60, 200, 110)

FIREWORK_COLORS = [
    (255, 50, 50),
    (255, 120, 30),
    (255, 220, 40),
    (255, 255, 120),
    (50, 255, 100),
    (40, 220, 255),
    (80, 140, 255),
    (180, 80, 255),
    (255, 80, 200),
    (255, 180, 220),
    (255, 255, 255),
]

EDGE_RADIUS = 0.002
KNOWN_EDGE_RADIUS = 0.005
NODE_RADIUS_GREY = 0.2
NODE_RADIUS_OBSERVED = 0.5
NODE_RADIUS_VISITED = 0.6

TRAIL_COLOR = (220, 45, 45)
TRAIL_RADIUS = 0.015

STEPS_RATE_WINDOW_SEC = 5.0


def blue_ramp(t: float) -> tuple[int, int, int]:
    """Interpolate between dark blue (t=0) and light blue (t=1)."""
    t = max(0.0, min(1.0, t))
    return tuple(
        int(round(dark + (light - dark) * t)) for dark, light in zip(DARK_BLUE, LIGHT_BLUE)
    )
