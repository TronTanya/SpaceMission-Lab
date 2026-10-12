"""
Орбитальная механика: идеальный перелёт Хомана между круговыми компланарными орбитами.

Образовательная модель точечной массы со сферической Землёй. Не предназначена
для оперативного планирования полётов.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Константы (СИ)
# ---------------------------------------------------------------------------

# Геоцентрическая гравитационная постоянная GM (атмосфера включена).
# Источник: WGS 84 / IERS Numerical Standards, 3.986004418e14 m³/s².
MU_EARTH: float = 3.986004418e14

# Опорный радиус Земли для перевода высоты в геоцентрическое расстояние.
# Приближённый объёмный средний радиус (~6371 км), не экваториальный WGS 84.
R_EARTH: float = 6_371_000.0

# Стандартное ускорение свободного падения (для уравнения Циолковского).
G0: float = 9.80665

MODEL_VERSION: str = "1.0.0"

ALTITUDE_MIN_KM: float = 200.0
ALTITUDE_MAX_KM: float = 50_000.0


class OrbitalError(ValueError):
    """Ошибка валидации или расчёта орбитальных параметров."""


@dataclass(frozen=True)
class HohmannResult:
    """Результаты идеального перелёта Хомана."""

    h1_km: float
    h2_km: float
    r1_m: float
    r2_m: float
    a_transfer_m: float
    v1_mps: float
    v2_mps: float
    vt1_mps: float
    vt2_mps: float
    dv1_signed_mps: float
    dv2_signed_mps: float
    dv1_abs_mps: float
    dv2_abs_mps: float
    dv_total_mps: float
    t_transfer_s: float
    period1_s: float
    period2_s: float
    no_transfer: bool
    is_raising: bool
    is_lowering: bool
    direction1: str
    direction2: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_finite(name: str, value: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise OrbitalError(f"Параметр «{name}» должен быть числом.")
    value_f = float(value)
    if not math.isfinite(value_f):
        raise OrbitalError(f"Параметр «{name}» должен быть конечным числом.")
    return value_f


def validate_altitude_km(altitude_km: float, name: str = "высота орбиты") -> float:
    """Проверка высоты круговой орбиты в километрах."""
    value = _require_finite(name, altitude_km)
    if value < ALTITUDE_MIN_KM or value > ALTITUDE_MAX_KM:
        raise OrbitalError(
            f"Параметр «{name}» должен быть в диапазоне "
            f"{ALTITUDE_MIN_KM:.0f}–{ALTITUDE_MAX_KM:.0f} км "
            f"(получено {value})."
        )
    return value


def altitude_km_to_radius_m(altitude_km: float) -> float:
    """Перевод высоты над поверхностью (км) в геоцентрический радиус (м)."""
    h_km = validate_altitude_km(altitude_km)
    return R_EARTH + h_km * 1000.0


def circular_velocity(radius_m: float, mu: float = MU_EARTH) -> float:
    """Круговая скорость v = sqrt(μ / r)."""
    r = _require_finite("радиус", radius_m)
    if r <= 0:
        raise OrbitalError("Радиус орбиты должен быть положительным.")
    return math.sqrt(mu / r)


def orbital_period(radius_m: float, mu: float = MU_EARTH) -> float:
    """Период круговой орбиты T = 2π √(r³ / μ)."""
    r = _require_finite("радиус", radius_m)
    if r <= 0:
        raise OrbitalError("Радиус орбиты должен быть положительным.")
    return 2.0 * math.pi * math.sqrt((r ** 3) / mu)


def vis_viva(radius_m: float, semi_major_m: float, mu: float = MU_EARTH) -> float:
    """Уравнение вис-вива: v = sqrt(μ (2/r − 1/a))."""
    r = _require_finite("радиус", radius_m)
    a = _require_finite("большая полуось", semi_major_m)
    if r <= 0 or a <= 0:
        raise OrbitalError("Радиус и большая полуось должны быть положительными.")
    arg = 2.0 / r - 1.0 / a
    if arg < 0:
        raise OrbitalError("Аргумент уравнения вис-вива отрицателен.")
    return math.sqrt(mu * arg)


def _maneuver_direction(signed_dv: float) -> str:
    if abs(signed_dv) < 1e-12:
        return "не требуется"
    if signed_dv > 0:
        return "ускорение (по направлению скорости)"
    return "торможение (против направления скорости)"


def calculate_hohmann(
    h1_km: float,
    h2_km: float,
    mu: float = MU_EARTH,
) -> HohmannResult:
    """
    Расчёт идеального перелёта Хомана между двумя круговыми компланарными орбитами.

    При равных высотах перелёт не требуется: Δv и время перелёта равны нулю.
    """
    h1 = validate_altitude_km(h1_km, "высота начальной орбиты")
    h2 = validate_altitude_km(h2_km, "высота целевой орбиты")

    r1 = R_EARTH + h1 * 1000.0
    r2 = R_EARTH + h2 * 1000.0

    v1 = circular_velocity(r1, mu)
    v2 = circular_velocity(r2, mu)
    t1 = orbital_period(r1, mu)
    t2 = orbital_period(r2, mu)

    if abs(h1 - h2) < 1e-12:
        return HohmannResult(
            h1_km=h1,
            h2_km=h2,
            r1_m=r1,
            r2_m=r2,
            a_transfer_m=r1,
            v1_mps=v1,
            v2_mps=v2,
            vt1_mps=v1,
            vt2_mps=v2,
            dv1_signed_mps=0.0,
            dv2_signed_mps=0.0,
            dv1_abs_mps=0.0,
            dv2_abs_mps=0.0,
            dv_total_mps=0.0,
            t_transfer_s=0.0,
            period1_s=t1,
            period2_s=t2,
            no_transfer=True,
            is_raising=False,
            is_lowering=False,
            direction1="не требуется",
            direction2="не требуется",
        )

    a = (r1 + r2) / 2.0
    vt1 = vis_viva(r1, a, mu)
    vt2 = vis_viva(r2, a, mu)

    # Знаковые касательные импульсы (положительный — ускорение).
    dv1_signed = vt1 - v1
    dv2_signed = v2 - vt2
    dv_total = abs(dv1_signed) + abs(dv2_signed)
    t_transfer = math.pi * math.sqrt((a ** 3) / mu)

    is_raising = r2 > r1
    is_lowering = r2 < r1

    return HohmannResult(
        h1_km=h1,
        h2_km=h2,
        r1_m=r1,
        r2_m=r2,
        a_transfer_m=a,
        v1_mps=v1,
        v2_mps=v2,
        vt1_mps=vt1,
        vt2_mps=vt2,
        dv1_signed_mps=dv1_signed,
        dv2_signed_mps=dv2_signed,
        dv1_abs_mps=abs(dv1_signed),
        dv2_abs_mps=abs(dv2_signed),
        dv_total_mps=dv_total,
        t_transfer_s=t_transfer,
        period1_s=t1,
        period2_s=t2,
        no_transfer=False,
        is_raising=is_raising,
        is_lowering=is_lowering,
        direction1=_maneuver_direction(dv1_signed),
        direction2=_maneuver_direction(dv2_signed),
    )


def specific_orbital_energy(radius_m: float, speed_mps: float, mu: float = MU_EARTH) -> float:
    """Удельная орбитальная энергия ε = v²/2 − μ/r."""
    r = _require_finite("радиус", radius_m)
    v = _require_finite("скорость", speed_mps)
    if r <= 0:
        raise OrbitalError("Радиус должен быть положительным.")
    return 0.5 * v * v - mu / r
