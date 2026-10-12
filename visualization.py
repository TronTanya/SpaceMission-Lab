"""
Геометрия перелётного эллипса, решение уравнения Кеплера и построение схемы орбит.

Положение аппарата определяется по прошедшему времени через уравнение Кеплера
M = E − e·sin E, а не равномерным движением по углу.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go

from orbital import MU_EARTH, R_EARTH, HohmannResult

KEPLER_TOL: float = 1e-12
KEPLER_MAX_ITER: int = 50

COLOR_TEXT = "#202733"
COLOR_MUTED = "#667085"
COLOR_BORDER = "#E4E7EC"
COLOR_ACCENT = "#315EFB"
COLOR_SUCCESS = "#278451"
COLOR_WARNING = "#B7791F"
COLOR_EARTH_FILL = "#DCE4F2"
COLOR_EARTH_LINE = "#98A2B3"


class KeplerError(RuntimeError):
    """Решатель уравнения Кеплера не сошёлся."""


@dataclass(frozen=True)
class KeplerSolution:
    eccentric_anomaly: float
    iterations: int
    converged: bool
    residual: float


@dataclass(frozen=True)
class TransferState:
    """Состояние аппарата на перелётной траектории."""

    fraction: float
    elapsed_s: float
    x_m: float
    y_m: float
    vx_mps: float
    vy_mps: float
    radius_m: float
    speed_mps: float
    true_anomaly_rad: float


def solve_kepler(
    mean_anomaly: float,
    eccentricity: float,
    tol: float = KEPLER_TOL,
    max_iter: int = KEPLER_MAX_ITER,
) -> KeplerSolution:
    """
    Решение уравнения Кеплера E − e·sin E = M для эллипса (0 ≤ e < 1).

    Метод Ньютона с защитной бисекцией: корень всегда лежит в [M − e, M + e],
    так как функция f(E) = E − e·sin E − M монотонно возрастает.
    """
    if not (math.isfinite(mean_anomaly) and math.isfinite(eccentricity)):
        raise ValueError("Средняя аномалия и эксцентриситет должны быть конечными.")
    if eccentricity < 0 or eccentricity >= 1:
        raise ValueError("Эксцентриситет должен быть в диапазоне [0, 1).")

    m = mean_anomaly
    e = eccentricity
    if e == 0.0:
        return KeplerSolution(m, 0, True, 0.0)

    lo, hi = m - e, m + e
    big_e = m if e < 0.8 else m + e * math.sin(m)
    big_e = min(max(big_e, lo), hi)

    residual = big_e - e * math.sin(big_e) - m
    if abs(residual) <= tol:
        return KeplerSolution(big_e, 0, True, abs(residual))
    for iteration in range(1, max_iter + 1):
        if residual > 0:
            hi = big_e
        else:
            lo = big_e
        derivative = 1.0 - e * math.cos(big_e)
        candidate = big_e - residual / derivative
        if not (lo < candidate < hi):
            candidate = 0.5 * (lo + hi)
        big_e = candidate
        residual = big_e - e * math.sin(big_e) - m
        if abs(residual) <= tol:
            return KeplerSolution(big_e, iteration, True, abs(residual))
        if hi - lo <= tol:
            return KeplerSolution(big_e, iteration, abs(residual) <= 1e-9, abs(residual))

    return KeplerSolution(big_e, max_iter, False, abs(residual))


def transfer_eccentricity(result: HohmannResult) -> float:
    if result.no_transfer:
        return 0.0
    return abs(result.r2_m - result.r1_m) / (result.r1_m + result.r2_m)


def transfer_state(
    result: HohmannResult,
    fraction: float,
    mu: float = MU_EARTH,
) -> TransferState:
    """
    Положение и скорость аппарата при доле прошедшего времени перелёта fraction ∈ [0, 1].

    Первый манёвр всегда в точке (r1, 0); движение — против часовой стрелки.
    При подъёме старт в перицентре, при снижении — в апоцентре.
    """
    if not math.isfinite(fraction):
        raise ValueError("Доля перелёта должна быть конечной.")
    f = min(max(float(fraction), 0.0), 1.0)

    if result.no_transfer:
        r = result.r1_m
        return TransferState(f, 0.0, r, 0.0, 0.0, result.v1_mps, r, result.v1_mps, 0.0)

    a = result.a_transfer_m
    e = transfer_eccentricity(result)
    b = a * math.sqrt(1.0 - e * e)
    start_offset = 0.0 if result.is_raising else math.pi
    mean_anomaly = start_offset + math.pi * f

    sol = solve_kepler(mean_anomaly, e)
    if not sol.converged:
        raise KeplerError(
            f"Уравнение Кеплера не сошлось (M={mean_anomaly}, e={e}, невязка={sol.residual})."
        )
    big_e = sol.eccentric_anomaly
    cos_e, sin_e = math.cos(big_e), math.sin(big_e)

    x_p = a * (cos_e - e)
    y_p = b * sin_e
    r = a * (1.0 - e * cos_e)
    factor = math.sqrt(mu * a) / r
    vx_p = -factor * sin_e
    vy_p = factor * math.sqrt(1.0 - e * e) * cos_e

    # При снижении перицентр направлен в сторону −x.
    sign = 1.0 if result.is_raising else -1.0
    nu = 2.0 * math.atan2(math.sqrt(1.0 + e) * math.sin(big_e / 2.0),
                          math.sqrt(1.0 - e) * math.cos(big_e / 2.0))

    return TransferState(
        fraction=f,
        elapsed_s=f * result.t_transfer_s,
        x_m=sign * x_p,
        y_m=sign * y_p,
        vx_mps=sign * vx_p,
        vy_mps=sign * vy_p,
        radius_m=r,
        speed_mps=math.hypot(vx_p, vy_p),
        true_anomaly_rad=nu,
    )


def sample_transfer(result: HohmannResult, n: int = 200) -> list[TransferState]:
    """Равномерная по времени выборка состояний вдоль перелёта."""
    if n < 2:
        raise ValueError("Число точек должно быть не меньше 2.")
    return [transfer_state(result, f) for f in np.linspace(0.0, 1.0, n)]


def _circle(radius_m: float, n: int = 361) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * math.pi, n)
    return radius_m * np.cos(theta), radius_m * np.sin(theta)


def _full_ellipse(result: HohmannResult, n: int = 361) -> tuple[np.ndarray, np.ndarray]:
    a = result.a_transfer_m
    e = transfer_eccentricity(result)
    b = a * math.sqrt(1.0 - e * e)
    big_e = np.linspace(0.0, 2.0 * math.pi, n)
    sign = 1.0 if result.is_raising else -1.0
    return sign * a * (np.cos(big_e) - e), sign * b * np.sin(big_e)


def build_orbit_figure(result: HohmannResult, progress: float = 0.0) -> go.Figure:
    """Схема орбит в километрах с равным масштабом осей."""
    km = 1e-3
    fig = go.Figure()

    ex, ey = _circle(R_EARTH, 181)
    fig.add_trace(go.Scatter(
        x=ex * km, y=ey * km, mode="lines", fill="toself",
        fillcolor=COLOR_EARTH_FILL, line=dict(color=COLOR_EARTH_LINE, width=1),
        name="Земля (R = 6371 км)", hoverinfo="name",
    ))

    x1, y1 = _circle(result.r1_m)
    fig.add_trace(go.Scatter(
        x=x1 * km, y=y1 * km, mode="lines",
        line=dict(color=COLOR_MUTED, width=1.5),
        name=f"Начальная орбита ({result.h1_km:,.0f} км)".replace(",", " "),
        hoverinfo="name",
    ))

    if not result.no_transfer:
        x2, y2 = _circle(result.r2_m)
        fig.add_trace(go.Scatter(
            x=x2 * km, y=y2 * km, mode="lines",
            line=dict(color=COLOR_SUCCESS, width=1.5),
            name=f"Целевая орбита ({result.h2_km:,.0f} км)".replace(",", " "),
            hoverinfo="name",
        ))

        fx, fy = _full_ellipse(result)
        fig.add_trace(go.Scatter(
            x=fx * km, y=fy * km, mode="lines",
            line=dict(color=COLOR_ACCENT, width=1, dash="dot"),
            name="Перелётный эллипс", hoverinfo="name",
        ))

        states = sample_transfer(result, 240)
        fig.add_trace(go.Scatter(
            x=[s.x_m * km for s in states], y=[s.y_m * km for s in states],
            mode="lines", line=dict(color=COLOR_ACCENT, width=2.5),
            name="Пройденная половина эллипса", hoverinfo="name",
        ))

        end = states[-1]
        fig.add_trace(go.Scatter(
            x=[result.r1_m * km, end.x_m * km], y=[0.0, end.y_m * km],
            mode="markers+text",
            marker=dict(color=COLOR_WARNING, size=10, symbol="diamond",
                        line=dict(color="#FFFFFF", width=1)),
            text=["Манёвр 1", "Манёвр 2"], textposition="bottom center",
            textfont=dict(color=COLOR_TEXT, size=12),
            name="Точки манёвров", hoverinfo="name",
        ))

    craft = transfer_state(result, progress)
    fig.add_trace(go.Scatter(
        x=[craft.x_m * km], y=[craft.y_m * km], mode="markers",
        marker=dict(color=COLOR_TEXT, size=11, line=dict(color="#FFFFFF", width=2)),
        name="Аппарат",
        hovertemplate=(
            f"Аппарат<br>r = {craft.radius_m * km:,.0f} км<br>"
            f"v = {craft.speed_mps:,.0f} м/с<extra></extra>"
        ).replace(",", " "),
    ))

    r_max = max(result.r1_m, result.r2_m) * km * 1.12
    fig.update_layout(
        height=600,
        margin=dict(l=10, r=10, t=70, b=10),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color=COLOR_TEXT, size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(range=[-r_max, r_max], title="x, км", gridcolor=COLOR_BORDER,
                   zerolinecolor=COLOR_BORDER),
        yaxis=dict(range=[-r_max, r_max], title="y, км", gridcolor=COLOR_BORDER,
                   zerolinecolor=COLOR_BORDER, scaleanchor="x", scaleratio=1),
        dragmode="zoom",
    )
    return fig
