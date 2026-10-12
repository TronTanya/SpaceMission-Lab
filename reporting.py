"""
Экспорт и импорт: JSON-конфигурация, CSV-результаты, автономный HTML-отчёт.

Импорт JSON строго валидируется; содержимое никогда не исполняется.
"""

from __future__ import annotations

import html
import json
import math
from datetime import datetime
from typing import Any

import pandas as pd

from experiments import MissionConfig, MissionResult, format_duration
from orbital import G0, MODEL_VERSION, MU_EARTH, R_EARTH

APP_NAME = "SpaceMission Lab"
SCHEMA_VERSION = 1
MAX_JSON_BYTES = 64 * 1024

CONFIG_FIELDS: dict[str, str] = {
    "initial_altitude_km": "км",
    "target_altitude_km": "км",
    "dry_mass_kg": "кг",
    "propellant_kg": "кг",
    "isp_s": "с",
}

CONFIG_LABELS: dict[str, str] = {
    "initial_altitude_km": "Высота начальной орбиты",
    "target_altitude_km": "Высота целевой орбиты",
    "dry_mass_kg": "Сухая масса аппарата",
    "propellant_kg": "Доступный запас топлива",
    "isp_s": "Удельный импульс",
}

EQUATIONS: list[tuple[str, str]] = [
    ("Радиус орбиты", "r = R⊕ + h"),
    ("Круговая скорость", "v = √(μ / r)"),
    ("Большая полуось перелётного эллипса", "a = (r1 + r2) / 2"),
    ("Уравнение вис-вива", "v = √(μ (2/r − 1/a))"),
    ("Импульсы", "Δv1 = vt1 − v1;  Δv2 = v2 − vt2;  Δv = |Δv1| + |Δv2|"),
    ("Время перелёта", "t = π √(a³ / μ)"),
    ("Период круговой орбиты", "T = 2π √(r³ / μ)"),
    ("Доступная Δv (Циолковский)", "Δv_дост = Isp · g0 · ln(m0 / m_сух)"),
    ("Конечная масса", "m_f = m0 · exp(−Δv / (Isp · g0))"),
    ("Минимальное топливо", "m_т,min = m_сух · (exp(Δv / (Isp · g0)) − 1)"),
]

LIMITATIONS: list[str] = [
    "Сферическая Земля как точечная масса; несферичность гравитационного поля не учитывается.",
    "Не учитывается атмосферное сопротивление.",
    "Не учитываются возмущения от Луны, Солнца и других тел.",
    "Только компланарные круговые орбиты; изменение наклонения не моделируется.",
    "Импульсные манёвры: конечная длительность работы двигателя не моделируется.",
    "Не учитываются ошибки навигации и исполнения манёвров.",
    "Не учитываются резервы топлива для других операций (ориентация, коррекции, увод).",
    "Перелёт Хомана выводит на целевую орбиту, но не обеспечивает встречу с другим аппаратом.",
    "Выполнимость по топливу не означает выполнимости миссии в целом.",
]

DISCLAIMER = (
    "Образовательная орбитальная модель. Не является программным обеспечением "
    "для оперативного планирования космических полётов."
)


class ConfigImportError(ValueError):
    """Ошибка импорта JSON-конфигурации."""


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def config_to_json(config: MissionConfig) -> str:
    payload = {
        "app": APP_NAME,
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "units": CONFIG_FIELDS,
        "config": config.to_dict(),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _reject_constant(token: str) -> Any:
    raise ConfigImportError(f"Недопустимое значение в JSON: {token}.")


def parse_config_json(data: str | bytes) -> MissionConfig:
    """Строгий разбор JSON-конфигурации миссии."""
    if isinstance(data, bytes):
        if len(data) > MAX_JSON_BYTES:
            raise ConfigImportError("Файл слишком большой.")
        try:
            data = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ConfigImportError("Файл должен быть в кодировке UTF-8.") from exc
    if not isinstance(data, str):
        raise ConfigImportError("Ожидается текст JSON.")
    if len(data.encode("utf-8")) > MAX_JSON_BYTES:
        raise ConfigImportError("Файл слишком большой.")

    try:
        payload = json.loads(data, parse_constant=_reject_constant)
    except ConfigImportError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ConfigImportError("Некорректный формат JSON.") from exc

    if not isinstance(payload, dict):
        raise ConfigImportError("Корневой элемент JSON должен быть объектом.")

    version = payload.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise ConfigImportError("Поле «schema_version» отсутствует или не является целым числом.")
    if version != SCHEMA_VERSION:
        raise ConfigImportError(f"Неподдерживаемая версия схемы: {version}.")

    config = payload.get("config")
    if not isinstance(config, dict):
        raise ConfigImportError("Поле «config» должно быть объектом.")

    missing = [k for k in CONFIG_FIELDS if k not in config]
    if missing:
        raise ConfigImportError(f"Отсутствуют поля: {', '.join(missing)}.")
    unknown = [k for k in config if k not in CONFIG_FIELDS]
    if unknown:
        raise ConfigImportError(f"Неизвестные поля: {', '.join(map(str, unknown))}.")

    values: dict[str, float] = {}
    for key in CONFIG_FIELDS:
        value = config[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ConfigImportError(f"Поле «{key}» должно быть числом.")
        try:
            number = float(value)
        except OverflowError as exc:
            raise ConfigImportError(f"Поле «{key}» вне допустимого диапазона.") from exc
        if not math.isfinite(number):
            raise ConfigImportError(f"Поле «{key}» должно быть конечным числом.")
        values[key] = number

    try:
        return MissionConfig(**values).validated()
    except ValueError as exc:
        raise ConfigImportError(str(exc)) from exc


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """CSV в UTF-8 с BOM (корректно открывается в Excel с кириллицей)."""
    return df.to_csv(index=False).encode("utf-8-sig")


def _fuel_status(mission: MissionResult) -> str:
    t, p = mission.transfer, mission.propellant
    if t.no_transfer:
        return "Перелёт не требуется"
    return "Выполним по топливу" if p.feasible else "Невыполним при введённом запасе топлива"


def results_rows(mission: MissionResult) -> list[tuple[str, str, str]]:
    """Строки (параметр, значение, единица) для таблиц, CSV и отчёта."""
    c, t, p = mission.config, mission.transfer, mission.propellant
    rows: list[tuple[str, str, str]] = []
    for key, unit in CONFIG_FIELDS.items():
        rows.append((CONFIG_LABELS[key], f"{getattr(c, key):.6g}", unit))
    rows += [
        ("Радиус начальной орбиты r1", f"{t.r1_m / 1000:.3f}", "км"),
        ("Радиус целевой орбиты r2", f"{t.r2_m / 1000:.3f}", "км"),
        ("Большая полуось перелётного эллипса a", f"{t.a_transfer_m / 1000:.3f}", "км"),
        ("Круговая скорость v1", f"{t.v1_mps:.3f}", "м/с"),
        ("Круговая скорость v2", f"{t.v2_mps:.3f}", "м/с"),
        ("Скорость на эллипсе в точке 1 vt1", f"{t.vt1_mps:.3f}", "м/с"),
        ("Скорость на эллипсе в точке 2 vt2", f"{t.vt2_mps:.3f}", "м/с"),
        ("Импульс 1 (со знаком)", f"{t.dv1_signed_mps:.3f}", "м/с"),
        ("Направление импульса 1", t.direction1, ""),
        ("Импульс 2 (со знаком)", f"{t.dv2_signed_mps:.3f}", "м/с"),
        ("Направление импульса 2", t.direction2, ""),
        ("Суммарный Δv", f"{t.dv_total_mps:.3f}", "м/с"),
        ("Время перелёта", f"{t.t_transfer_s:.1f}", "с"),
        ("Период начальной орбиты", f"{t.period1_s:.1f}", "с"),
        ("Период целевой орбиты", f"{t.period2_s:.1f}", "с"),
        ("Скорость истечения Isp·g0", f"{p.exhaust_velocity_mps:.3f}", "м/с"),
        ("Начальная масса m0", f"{p.initial_mass_kg:.3f}", "кг"),
        ("Доступная Δv", f"{p.dv_available_mps:.3f}", "м/с"),
        ("Минимальное топливо для сухого аппарата", f"{p.minimum_propellant_kg:.3f}", "кг"),
        (
            "Расход топлива при введённой заправке"
            if p.feasible else "Теоретическая потребность в топливе при введённой заправке",
            f"{p.propellant_required_kg:.3f}",
            "кг",
        ),
        ("Статус бюджета топлива", _fuel_status(mission), ""),
    ]
    if p.feasible:
        rows.append(("Остаток топлива", f"{p.remaining_propellant_kg:.3f}", "кг"))
        rows.append(("Запас Δv", f"{p.dv_margin_mps:.3f}", "м/с"))
    else:
        rows.append(("Дефицит Δv", f"{p.dv_shortfall_mps:.3f}", "м/с"))
    return rows


def results_dataframe(mission: MissionResult) -> pd.DataFrame:
    return pd.DataFrame(results_rows(mission), columns=["Параметр", "Значение", "Единица"])


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

def _table(rows: list[tuple[str, ...]], header: tuple[str, ...]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in header)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def build_html_report(mission: MissionResult, generated_at: datetime | None = None) -> str:
    """Автономный HTML-отчёт; все значения экранируются."""
    ts = (generated_at or datetime.now()).strftime("%Y-%m-%d %H:%M")
    t, p = mission.transfer, mission.propellant
    status = _fuel_status(mission)
    status_color = "#278451" if p.feasible else "#B7791F"

    constants = [
        ("Гравитационный параметр Земли μ", f"{MU_EARTH:.9e}", "м³/с²"),
        ("Опорный радиус Земли R⊕", f"{R_EARTH:.0f}", "м"),
        ("Стандартное ускорение g0", f"{G0}", "м/с²"),
    ]
    summary = [
        ("Суммарный Δv", f"{t.dv_total_mps:.1f} м/с"),
        ("Время перелёта", format_duration(t.t_transfer_s)),
        ("Минимальное топливо", f"{p.minimum_propellant_kg:.1f} кг"),
        ("Доступная Δv", f"{p.dv_available_mps:.1f} м/с"),
    ]

    css = (
        "body{font-family:-apple-system,'Segoe UI',Roboto,Arial,sans-serif;color:#202733;"
        "background:#F7F8FA;margin:0;padding:24px}"
        "main{max-width:1000px;margin:0 auto;background:#FFF;border:1px solid #E4E7EC;"
        "border-radius:6px;padding:28px}"
        "h1{margin:0 0 4px;font-size:24px}h2{font-size:17px;margin:28px 0 8px}"
        ".muted{color:#667085;font-size:13px}"
        "table{border-collapse:collapse;width:100%;font-size:13px}"
        "th,td{border:1px solid #E4E7EC;padding:6px 8px;text-align:left}"
        "th{background:#F7F8FA}"
        ".note{border-left:3px solid #B7791F;padding:8px 12px;background:#FFF;font-size:13px}"
        ".status{font-weight:600}"
    )

    parts = [
        "<!DOCTYPE html><html lang='ru'><head><meta charset='utf-8'>",
        f"<title>{html.escape(APP_NAME)} — отчёт</title><style>{css}</style></head><body><main>",
        f"<h1>{html.escape(APP_NAME)}</h1>",
        "<div class='muted'>Отчёт об идеальном перелёте Хомана · версия модели "
        f"{html.escape(MODEL_VERSION)} · сформирован {html.escape(ts)}</div>",
        f"<p class='note'>{html.escape(DISCLAIMER)}</p>",
        "<h2>Сводка</h2>",
        _table(summary, ("Показатель", "Значение")),
        f"<p class='status' style='color:{status_color}'>Статус бюджета топлива: "
        f"{html.escape(status)}</p>",
        "<h2>Входные параметры и результаты</h2>",
        _table(results_rows(mission), ("Параметр", "Значение", "Единица")),
        "<h2>Константы</h2>",
        _table(constants, ("Константа", "Значение", "Единица")),
        "<h2>Уравнения</h2>",
        _table(EQUATIONS, ("Величина", "Формула")),
        "<h2>Ограничения модели</h2><ul>",
        "".join(f"<li>{html.escape(item)}</li>" for item in LIMITATIONS),
        "</ul></main></body></html>",
    ]
    return "".join(parts)
