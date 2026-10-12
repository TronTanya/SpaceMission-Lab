"""
Расчёт топлива по идеальному уравнению Циолковского.

Модель импульсная: не учитывает тягу, длительность включений, гравитационные
потери и резервы топлива для других операций.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from orbital import G0

DRY_MASS_MIN_KG: float = 1.0
DRY_MASS_MAX_KG: float = 100_000.0
PROPELLANT_MIN_KG: float = 0.0
PROPELLANT_MAX_KG: float = 100_000.0
ISP_MIN_S: float = 50.0
ISP_MAX_S: float = 5_000.0

# Допуск для сравнения требуемой и доступной Δv (м/с).
FEASIBILITY_TOLERANCE_MPS: float = 1e-9


class PropulsionError(ValueError):
    """Ошибка валидации параметров двигательной установки."""


@dataclass(frozen=True)
class PropellantResult:
    """Результаты расчёта бюджета топлива."""

    dry_mass_kg: float
    propellant_available_kg: float
    isp_s: float
    exhaust_velocity_mps: float
    initial_mass_kg: float
    dv_required_mps: float
    dv_available_mps: float
    final_mass_kg: float
    propellant_required_kg: float
    minimum_propellant_kg: float
    feasible: bool
    remaining_propellant_kg: float | None
    dv_margin_mps: float | None
    dv_shortfall_mps: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _check_range(name: str, value: float, low: float, high: float, unit: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise PropulsionError(f"Параметр «{name}» должен быть числом.")
    v = float(value)
    if not math.isfinite(v):
        raise PropulsionError(f"Параметр «{name}» должен быть конечным числом.")
    if v < low or v > high:
        raise PropulsionError(
            f"Параметр «{name}» должен быть в диапазоне {low:g}–{high:g} {unit} (получено {v:g})."
        )
    return v


def validate_dry_mass(value: float) -> float:
    return _check_range("сухая масса", value, DRY_MASS_MIN_KG, DRY_MASS_MAX_KG, "кг")


def validate_propellant(value: float) -> float:
    return _check_range("запас топлива", value, PROPELLANT_MIN_KG, PROPELLANT_MAX_KG, "кг")


def validate_isp(value: float) -> float:
    return _check_range("удельный импульс", value, ISP_MIN_S, ISP_MAX_S, "с")


def _validate_dv(dv_mps: float) -> float:
    if not isinstance(dv_mps, (int, float)) or isinstance(dv_mps, bool):
        raise PropulsionError("Δv должна быть числом.")
    dv = float(dv_mps)
    if not math.isfinite(dv) or dv < 0:
        raise PropulsionError("Δv должна быть конечным неотрицательным числом.")
    return dv


def exhaust_velocity(isp_s: float) -> float:
    """Эффективная скорость истечения ve = Isp · g0."""
    return validate_isp(isp_s) * G0


def available_delta_v(dry_mass_kg: float, propellant_kg: float, isp_s: float) -> float:
    """Доступная Δv: Isp · g0 · ln(m0 / m_dry)."""
    md = validate_dry_mass(dry_mass_kg)
    mp = validate_propellant(propellant_kg)
    ve = exhaust_velocity(isp_s)
    return ve * math.log((md + mp) / md)


def minimum_propellant(dry_mass_kg: float, dv_mps: float, isp_s: float) -> float:
    """Минимальное топливо для сухого аппарата: m_dry · (exp(Δv / ve) − 1)."""
    md = validate_dry_mass(dry_mass_kg)
    dv = _validate_dv(dv_mps)
    ve = exhaust_velocity(isp_s)
    return md * math.expm1(dv / ve)


def calculate_propellant(
    dry_mass_kg: float,
    propellant_kg: float,
    isp_s: float,
    dv_required_mps: float,
) -> PropellantResult:
    """
    Бюджет топлива для заданной Δv.

    Если конечная масса меньше сухой, перелёт невыполним при введённом запасе:
    расход возвращается как теоретическая потребность, а остаток — None.
    """
    md = validate_dry_mass(dry_mass_kg)
    mp = validate_propellant(propellant_kg)
    isp = validate_isp(isp_s)
    dv = _validate_dv(dv_required_mps)

    ve = isp * G0
    m0 = md + mp
    dv_avail = ve * math.log(m0 / md)
    mf = m0 * math.exp(-dv / ve)
    prop_required = m0 - mf
    min_prop = md * math.expm1(dv / ve)

    feasible = dv <= dv_avail + FEASIBILITY_TOLERANCE_MPS

    if feasible:
        remaining = max(mp - prop_required, 0.0)
        margin = max(dv_avail - dv, 0.0)
        shortfall = None
    else:
        remaining = None
        margin = None
        shortfall = dv - dv_avail

    return PropellantResult(
        dry_mass_kg=md,
        propellant_available_kg=mp,
        isp_s=isp,
        exhaust_velocity_mps=ve,
        initial_mass_kg=m0,
        dv_required_mps=dv,
        dv_available_mps=dv_avail,
        final_mass_kg=mf,
        propellant_required_kg=prop_required,
        minimum_propellant_kg=min_prop,
        feasible=feasible,
        remaining_propellant_kg=remaining,
        dv_margin_mps=margin,
        dv_shortfall_mps=shortfall,
    )
