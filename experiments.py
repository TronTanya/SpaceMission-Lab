"""
Конфигурация миссии, демонстрационные пресеты и воспроизводимые параметрические эксперименты.

Все результаты вычисляются единым расчётным ядром (orbital + propulsion).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from orbital import HohmannResult, calculate_hohmann, validate_altitude_km
from propulsion import (
    PropellantResult,
    calculate_propellant,
    validate_dry_mass,
    validate_isp,
    validate_propellant,
)


@dataclass(frozen=True)
class MissionConfig:
    """Входные параметры миссии (пользовательские единицы: км, кг, с)."""

    initial_altitude_km: float
    target_altitude_km: float
    dry_mass_kg: float
    propellant_kg: float
    isp_s: float

    def validated(self) -> "MissionConfig":
        return MissionConfig(
            initial_altitude_km=validate_altitude_km(self.initial_altitude_km, "высота начальной орбиты"),
            target_altitude_km=validate_altitude_km(self.target_altitude_km, "высота целевой орбиты"),
            dry_mass_kg=validate_dry_mass(self.dry_mass_kg),
            propellant_kg=validate_propellant(self.propellant_kg),
            isp_s=validate_isp(self.isp_s),
        )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class MissionResult:
    config: MissionConfig
    transfer: HohmannResult
    propellant: PropellantResult


def run_mission(config: MissionConfig) -> MissionResult:
    cfg = config.validated()
    transfer = calculate_hohmann(cfg.initial_altitude_km, cfg.target_altitude_km)
    prop = calculate_propellant(cfg.dry_mass_kg, cfg.propellant_kg, cfg.isp_s, transfer.dv_total_mps)
    return MissionResult(cfg, transfer, prop)


# Пресеты содержат только входные параметры; результаты считаются при запуске.
PRESETS: dict[str, MissionConfig] = {
    "Низкая орбита: 400 → 800 км": MissionConfig(400.0, 800.0, 500.0, 150.0, 300.0),
    "Высокая орбита: 400 → 35 786 км": MissionConfig(400.0, 35_786.0, 1_500.0, 4_000.0, 320.0),
    "Снижение орбиты: 1 000 → 400 км": MissionConfig(1_000.0, 400.0, 500.0, 100.0, 300.0),
}

# ---------------------------------------------------------------------------
# Параметрические эксперименты
# ---------------------------------------------------------------------------

SWEEP_INITIAL_ALTITUDE_KM: float = 400.0
SWEEP_TARGET_MIN_KM: float = 500.0
SWEEP_TARGET_MAX_KM: float = 50_000.0
SWEEP_POINTS: int = 100
SWEEP_DRY_MASS_KG: float = 1_000.0
SWEEP_ISP_S: float = 320.0

ISP_SWEEP_MIN_S: float = 200.0
ISP_SWEEP_MAX_S: float = 4_500.0
ISP_SWEEP_POINTS: int = 60

COL_TARGET_ALT = "Высота целевой орбиты, км"
COL_DV = "Суммарный Δv, м/с"
COL_TIME = "Время перелёта, ч"
COL_MIN_PROP = "Минимальное топливо, кг"
COL_ISP = "Удельный импульс, с"


def altitude_sweep(
    initial_altitude_km: float = SWEEP_INITIAL_ALTITUDE_KM,
    target_min_km: float = SWEEP_TARGET_MIN_KM,
    target_max_km: float = SWEEP_TARGET_MAX_KM,
    points: int = SWEEP_POINTS,
    dry_mass_kg: float = SWEEP_DRY_MASS_KG,
    isp_s: float = SWEEP_ISP_S,
) -> pd.DataFrame:
    """Эксперимент 1: изменение высоты целевой орбиты при фиксированных m_dry и Isp."""
    if points < 2:
        raise ValueError("Число точек должно быть не меньше 2.")
    validate_altitude_km(target_min_km)
    validate_altitude_km(target_max_km)
    if target_min_km >= target_max_km:
        raise ValueError("Нижняя граница диапазона должна быть меньше верхней.")

    rows: list[dict[str, Any]] = []
    for h2 in np.linspace(target_min_km, target_max_km, points):
        transfer = calculate_hohmann(initial_altitude_km, float(h2))
        prop = calculate_propellant(dry_mass_kg, 0.0, isp_s, transfer.dv_total_mps)
        rows.append({
            COL_TARGET_ALT: float(h2),
            COL_DV: transfer.dv_total_mps,
            COL_TIME: transfer.t_transfer_s / 3600.0,
            COL_MIN_PROP: prop.minimum_propellant_kg,
        })
    return pd.DataFrame(rows)


def isp_sweep(
    initial_altitude_km: float = 400.0,
    target_altitude_km: float = 35_786.0,
    dry_mass_kg: float = SWEEP_DRY_MASS_KG,
    isp_min_s: float = ISP_SWEEP_MIN_S,
    isp_max_s: float = ISP_SWEEP_MAX_S,
    points: int = ISP_SWEEP_POINTS,
) -> pd.DataFrame:
    """Эксперимент 2: изменение Isp при фиксированном орбитальном перелёте."""
    if points < 2:
        raise ValueError("Число точек должно быть не меньше 2.")
    validate_isp(isp_min_s)
    validate_isp(isp_max_s)
    if isp_min_s >= isp_max_s:
        raise ValueError("Нижняя граница диапазона должна быть меньше верхней.")

    transfer = calculate_hohmann(initial_altitude_km, target_altitude_km)
    rows: list[dict[str, Any]] = []
    for isp in np.linspace(isp_min_s, isp_max_s, points):
        prop = calculate_propellant(dry_mass_kg, 0.0, float(isp), transfer.dv_total_mps)
        rows.append({
            COL_ISP: float(isp),
            COL_DV: transfer.dv_total_mps,
            COL_MIN_PROP: prop.minimum_propellant_kg,
        })
    return pd.DataFrame(rows)


def format_duration(seconds: float) -> str:
    """Время в минутах (до 2 ч) или часах."""
    if not math.isfinite(seconds):
        return "—"
    if seconds == 0:
        return "0 мин"
    if seconds < 7200:
        return f"{seconds / 60:.1f} мин"
    return f"{seconds / 3600:.2f} ч"


def format_dv(mps: float) -> str:
    """Δv в м/с (до 10 км/с) или км/с."""
    if not math.isfinite(mps):
        return "—"
    if abs(mps) < 10_000:
        return f"{mps:,.1f} м/с".replace(",", " ")
    return f"{mps / 1000:.3f} км/с"


def format_kg(kg: float) -> str:
    if not math.isfinite(kg):
        return "—"
    return f"{kg:,.1f} кг".replace(",", " ")
