"""
SpaceMission Lab — виртуальная лаборатория орбитальных перелётов.

Запуск: streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from experiments import (
    COL_DV,
    COL_ISP,
    COL_MIN_PROP,
    COL_TARGET_ALT,
    COL_TIME,
    ISP_SWEEP_MAX_S,
    ISP_SWEEP_MIN_S,
    PRESETS,
    SWEEP_DRY_MASS_KG,
    SWEEP_INITIAL_ALTITUDE_KM,
    SWEEP_ISP_S,
    SWEEP_POINTS,
    SWEEP_TARGET_MAX_KM,
    SWEEP_TARGET_MIN_KM,
    MissionConfig,
    MissionResult,
    altitude_sweep,
    format_duration,
    format_dv,
    format_kg,
    isp_sweep,
    run_mission,
)
from orbital import ALTITUDE_MAX_KM, ALTITUDE_MIN_KM, G0, MODEL_VERSION, MU_EARTH, R_EARTH
from propulsion import (
    DRY_MASS_MAX_KG,
    DRY_MASS_MIN_KG,
    ISP_MAX_S,
    ISP_MIN_S,
    PROPELLANT_MAX_KG,
    PROPELLANT_MIN_KG,
)
from reporting import (
    DISCLAIMER,
    EQUATIONS,
    LIMITATIONS,
    ConfigImportError,
    build_html_report,
    config_to_json,
    dataframe_to_csv_bytes,
    parse_config_json,
    results_dataframe,
)
from visualization import (
    COLOR_ACCENT,
    COLOR_BORDER,
    COLOR_MUTED,
    COLOR_TEXT,
    build_orbit_figure,
    transfer_state,
)

st.set_page_config(page_title="SpaceMission Lab", layout="wide")

INPUT_KEYS: dict[str, str] = {
    "initial_altitude_km": "in_h1",
    "target_altitude_km": "in_h2",
    "dry_mass_kg": "in_dry",
    "propellant_kg": "in_prop",
    "isp_s": "in_isp",
}
DEFAULT_PRESET = "Низкая орбита: 400 → 800 км"
SCENARIO_LABELS = {"base": "Базовый сценарий", "modified": "Изменённый сценарий"}


# ---------------------------------------------------------------------------
# Состояние
# ---------------------------------------------------------------------------

def _set_inputs(config: MissionConfig) -> None:
    for field, key in INPUT_KEYS.items():
        st.session_state[key] = float(getattr(config, field))


def _init_state() -> None:
    if "initialized" not in st.session_state:
        _set_inputs(PRESETS[DEFAULT_PRESET])
        st.session_state.mission_config = None
        st.session_state.scenarios = {}
        st.session_state.import_message = None
        st.session_state.progress = 0
        st.session_state.initialized = True


def _current_config() -> MissionConfig:
    return MissionConfig(**{field: float(st.session_state[key]) for field, key in INPUT_KEYS.items()})


def _apply_preset() -> None:
    _set_inputs(PRESETS[st.session_state.preset_choice])
    st.session_state.mission_config = None


def _import_config() -> None:
    uploaded = st.session_state.get("config_upload")
    if uploaded is None:
        st.session_state.import_message = ("error", "Сначала выберите JSON-файл.")
        return
    try:
        config = parse_config_json(uploaded.getvalue())
    except ConfigImportError as exc:
        st.session_state.import_message = ("error", f"Конфигурация отклонена: {exc}")
        return
    _set_inputs(config)
    st.session_state.mission_config = None
    st.session_state.import_message = ("success", "Конфигурация загружена. Выполните расчёт.")


def _save_scenario(slot: str) -> None:
    cfg = st.session_state.mission_config
    if cfg is not None:
        st.session_state.scenarios[slot] = cfg


# ---------------------------------------------------------------------------
# Кешированные эксперименты
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _cached_altitude_sweep() -> pd.DataFrame:
    return altitude_sweep()


@st.cache_data(show_spinner=False)
def _cached_isp_sweep(h1: float, h2: float, dry: float) -> pd.DataFrame:
    return isp_sweep(initial_altitude_km=h1, target_altitude_km=h2, dry_mass_kg=dry)


def _line_chart(df: pd.DataFrame, x: str, y: str) -> go.Figure:
    fig = go.Figure(go.Scatter(x=df[x], y=df[y], mode="lines",
                               line=dict(color=COLOR_ACCENT, width=2), name=y))
    fig.update_layout(
        height=340, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(color=COLOR_TEXT, size=12),
        xaxis=dict(title=x, gridcolor=COLOR_BORDER),
        yaxis=dict(title=y, gridcolor=COLOR_BORDER, rangemode="tozero"),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Вкладка «Расчёт миссии»
# ---------------------------------------------------------------------------

def render_inputs() -> None:
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    c1.selectbox("Демонстрационный пресет", list(PRESETS), key="preset_choice")
    c2.button("Применить пресет", on_click=_apply_preset, width="stretch")
    st.caption(
        "Пресеты содержат только входные параметры и не воспроизводят реальные миссии. "
        "Высота 35 786 км сама по себе не делает орбиту геостационарной: "
        "нужны также нулевое наклонение и период, равный звёздным суткам."
    )

    left, right = st.columns(2, gap="large")
    with left, st.container(border=True):
        st.markdown("**Орбиты** · круговые, компланарные")
        st.number_input("Высота начальной орбиты, км", ALTITUDE_MIN_KM, ALTITUDE_MAX_KM,
                        step=50.0, format="%.1f", key="in_h1")
        st.number_input("Высота целевой орбиты, км", ALTITUDE_MIN_KM, ALTITUDE_MAX_KM,
                        step=50.0, format="%.1f", key="in_h2")
        st.caption("Высота отсчитывается от сферы радиусом 6371 км. "
                   "Допустимый диапазон: 200–50 000 км. Возможен подъём и снижение.")
    with right, st.container(border=True):
        st.markdown("**Аппарат и двигатель**")
        st.number_input("Сухая масса аппарата, кг", DRY_MASS_MIN_KG, DRY_MASS_MAX_KG,
                        step=10.0, format="%.1f", key="in_dry")
        st.number_input("Доступный запас топлива, кг", PROPELLANT_MIN_KG, PROPELLANT_MAX_KG,
                        step=10.0, format="%.1f", key="in_prop")
        st.number_input("Удельный импульс двигателя, с", ISP_MIN_S, ISP_MAX_S,
                        step=5.0, format="%.1f", key="in_isp")
        st.caption("Удельный импульс определяет расход топлива, но не тягу и не "
                   "длительность работы двигателя. Манёвры считаются мгновенными; для "
                   "двигателей малой тяги (например, электрореактивных) такое приближение "
                   "может быть непригодно.")


def render_metrics(mission: MissionResult) -> None:
    t, p = mission.transfer, mission.propellant
    cols = st.columns(4)
    cols[0].metric("Суммарный Δv", format_dv(t.dv_total_mps), border=True)
    cols[1].metric("Время перелёта", format_duration(t.t_transfer_s), border=True)
    cols[2].metric("Минимальный запас топлива", format_kg(p.minimum_propellant_kg), border=True,
                   help="Минимальная масса топлива для сухого аппарата по уравнению Циолковского.")
    if p.feasible:
        cols[3].metric("Запас Δv", format_dv(p.dv_margin_mps), border=True,
                       help="Доступная Δv минус требуемая.")
    else:
        cols[3].metric("Запас Δv", "нет", delta=f"дефицит {format_dv(p.dv_shortfall_mps)}",
                       delta_color="inverse", delta_arrow="off", border=True)


def render_status(mission: MissionResult) -> None:
    t, p = mission.transfer, mission.propellant
    if t.no_transfer:
        st.info("Высоты орбит совпадают: перелёт не требуется. Импульсы и время перелёта "
                "равны нулю, топливо не расходуется.")
    elif p.feasible:
        st.success(
            f"Перелёт выполним по введённому запасу топлива: расход "
            f"{format_kg(p.propellant_required_kg)}, остаток {format_kg(p.remaining_propellant_kg)}. "
            "Это проверка только бюджета топлива, а не выполнимости миссии в целом."
        )
    else:
        st.warning(
            f"Перелёт невыполним при введённом запасе топлива: доступно "
            f"{format_dv(p.dv_available_mps)}, требуется {format_dv(t.dv_total_mps)}. "
            f"Минимально необходимое топливо для сухого аппарата — "
            f"{format_kg(p.minimum_propellant_kg)}."
        )


def render_diagram(mission: MissionResult) -> None:
    t = mission.transfer
    st.markdown("#### Схема орбит")
    progress = st.slider("Прогресс перелёта, % времени", min_value=0, max_value=100, step=1,
                         key="progress", disabled=t.no_transfer)
    state = transfer_state(t, progress / 100.0)
    st.plotly_chart(build_orbit_figure(t, progress / 100.0), width="stretch",
                    config={"displaylogo": False, "scrollZoom": True})
    c1, c2, c3 = st.columns(3)
    c1.caption(f"Прошло времени: **{format_duration(state.elapsed_s)}**")
    c2.caption(f"Расстояние от центра Земли: **{state.radius_m / 1000:,.0f} км**".replace(",", " "))
    c3.caption(f"Скорость аппарата: **{state.speed_mps:,.0f} м/с**".replace(",", " "))
    st.caption("Положение аппарата вычисляется по прошедшему времени через уравнение Кеплера, "
               "поэтому у перигея аппарат движется быстрее, чем у апогея. Земля и орбиты "
               "изображены в едином масштабе; для высоких орбит используйте увеличение "
               "(выделение области или колесо мыши).")


def render_tables(mission: MissionResult) -> None:
    t, p = mission.transfer, mission.propellant
    st.markdown("#### Манёвры")
    if t.no_transfer:
        maneuvers = pd.DataFrame([
            {"Манёвр": "—", "Точка": "—", "Δv со знаком, м/с": 0.0, "|Δv|, м/с": 0.0,
             "Направление": "не требуется"},
        ])
    else:
        point1 = "перицентр перелётного эллипса" if t.is_raising else "апоцентр перелётного эллипса"
        point2 = "апоцентр перелётного эллипса" if t.is_raising else "перицентр перелётного эллипса"
        maneuvers = pd.DataFrame([
            {"Манёвр": "1 — сход с начальной орбиты", "Точка": point1,
             "Δv со знаком, м/с": round(t.dv1_signed_mps, 2), "|Δv|, м/с": round(t.dv1_abs_mps, 2),
             "Направление": t.direction1},
            {"Манёвр": "2 — выход на целевую орбиту", "Точка": point2,
             "Δv со знаком, м/с": round(t.dv2_signed_mps, 2), "|Δv|, м/с": round(t.dv2_abs_mps, 2),
             "Направление": t.direction2},
        ])
    st.dataframe(maneuvers, hide_index=True, width="stretch")

    orbit_rows = pd.DataFrame([
        {"Орбита": "Начальная", "Высота, км": t.h1_km, "Радиус, км": round(t.r1_m / 1000, 1),
         "Круговая скорость, м/с": round(t.v1_mps, 1), "Период": format_duration(t.period1_s)},
        {"Орбита": "Целевая", "Высота, км": t.h2_km, "Радиус, км": round(t.r2_m / 1000, 1),
         "Круговая скорость, м/с": round(t.v2_mps, 1), "Период": format_duration(t.period2_s)},
    ])
    st.dataframe(orbit_rows, hide_index=True, width="stretch")

    st.markdown("#### Бюджет топлива")
    consumption_label = ("Расход при введённой заправке" if p.feasible
                         else "Теоретическая потребность при введённой заправке")
    fuel = [
        ("Скорость истечения Isp·g0", format_dv(p.exhaust_velocity_mps)),
        ("Начальная масса m0 = m_сух + m_т", format_kg(p.initial_mass_kg)),
        ("Доступная Δv = Isp·g0·ln(m0/m_сух)", format_dv(p.dv_available_mps)),
        ("Требуемая Δv", format_dv(p.dv_required_mps)),
        ("Минимальное топливо для сухого аппарата", format_kg(p.minimum_propellant_kg)),
        (consumption_label, format_kg(p.propellant_required_kg)),
    ]
    if p.feasible:
        fuel += [("Остаток топлива", format_kg(p.remaining_propellant_kg)),
                 ("Запас Δv", format_dv(p.dv_margin_mps))]
    else:
        fuel += [("Остаток топлива", "отсутствует — запаса недостаточно"),
                 ("Дефицит Δv", format_dv(p.dv_shortfall_mps))]
    st.dataframe(pd.DataFrame(fuel, columns=["Величина", "Значение"]),
                 hide_index=True, width="stretch")

    with st.expander("Как читать бюджет топлива"):
        st.markdown(
            "- **Минимальное топливо** — сколько топлива нужно аппарату с указанной сухой массой, "
            "если заправить ровно столько, сколько требуется: m_сух·(exp(Δv/(Isp·g0)) − 1).\n"
            "- **Расход при введённой заправке** — сколько топлива сгорит, если стартовать с "
            "полным введённым запасом: m0·(1 − exp(−Δv/(Isp·g0))). При заправке сверх минимума "
            "расход больше минимального, потому что аппарат разгоняет и неизрасходованное "
            "топливо.\n"
            "- Если конечная масса оказывается меньше сухой, перелёт **невыполним** при "
            "введённом запасе, а расход показан как теоретическая потребность.\n"
            "- Выполнимость по топливу не означает выполнимости миссии: не учтены резервы, "
            "ориентация, ошибки навигации и другие операции."
        )
    with st.expander("Допущения модели"):
        st.markdown(
            "- Земля — сферическая точечная масса, μ = 3.986004418·10¹⁴ м³/с²; опорный радиус "
            "6 371 км используется только для перевода высоты в расстояние от центра.\n"
            "- Обе орбиты круговые и лежат в одной плоскости.\n"
            "- Два мгновенных касательных импульса (перелёт Хомана).\n"
            "- Нет атмосферы, возмущений и ошибок исполнения.\n"
            "- Перелёт выводит на целевую орбиту, но **не обеспечивает встречу** с другим "
            "аппаратом: для встречи нужен подбор момента старта (фазирование).\n"
            "- Для круговых компланарных орбит перелёт Хомана обычно экономичен по Δv, но он не "
            "является оптимальным среди всех возможных стратегий (например, при большом "
            "отношении радиусов орбит трёхимпульсный биэллиптический перелёт может "
            "потребовать меньше Δv)."
        )


def render_downloads(mission: MissionResult) -> None:
    st.markdown("#### Экспорт")
    c1, c2, c3 = st.columns(3)
    c1.download_button("Конфигурация (JSON)", config_to_json(mission.config).encode("utf-8"),
                       file_name="spacemission_config.json", mime="application/json",
                       width="stretch", on_click="ignore")
    c2.download_button("Результаты (CSV)", dataframe_to_csv_bytes(results_dataframe(mission)),
                       file_name="spacemission_results.csv", mime="text/csv",
                       width="stretch", on_click="ignore")
    c3.download_button("Отчёт (HTML)", build_html_report(mission).encode("utf-8"),
                       file_name="spacemission_report.html", mime="text/html",
                       width="stretch", on_click="ignore")


def render_import() -> None:
    with st.expander("Импорт конфигурации из JSON"):
        st.file_uploader("JSON-файл конфигурации", type=["json"], key="config_upload")
        st.button("Загрузить конфигурацию", on_click=_import_config)
        message = st.session_state.import_message
        if message:
            kind, text = message
            (st.success if kind == "success" else st.error)(text)


def tab_mission() -> None:
    render_inputs()
    render_import()

    try:
        current = _current_config().validated()
    except ValueError as exc:
        st.error(f"Некорректные входные данные: {exc}")
        st.session_state.mission_config = None
        return

    if st.button("Рассчитать перелёт", type="primary"):
        st.session_state.mission_config = current
        st.session_state.progress = 0

    saved = st.session_state.mission_config
    if saved is None:
        st.info("Задайте параметры и нажмите «Рассчитать перелёт».")
        return
    if saved != current:
        st.session_state.mission_config = None
        st.info("Параметры изменены — прежние результаты сброшены. Выполните расчёт заново.")
        return

    try:
        mission = run_mission(saved)
    except (ValueError, RuntimeError) as exc:
        st.error(f"Ошибка расчёта: {exc}")
        return

    st.divider()
    render_metrics(mission)
    render_status(mission)
    render_diagram(mission)
    render_tables(mission)
    render_downloads(mission)


# ---------------------------------------------------------------------------
# Вкладка «Сравнение сценариев»
# ---------------------------------------------------------------------------

def render_comparison() -> None:
    st.markdown("#### Сравнение двух сценариев")
    st.caption("Рассчитайте миссию на вкладке «Расчёт миссии», сохраните её как базовый "
               "сценарий, измените параметры, рассчитайте снова и сохраните как изменённый.")
    has_result = st.session_state.mission_config is not None
    c1, c2 = st.columns(2)
    c1.button("Сохранить текущий расчёт как «Базовый сценарий»", on_click=_save_scenario,
              args=("base",), disabled=not has_result, width="stretch")
    c2.button("Сохранить текущий расчёт как «Изменённый сценарий»", on_click=_save_scenario,
              args=("modified",), disabled=not has_result, width="stretch")
    if not has_result:
        st.caption("Нет актуального расчёта для сохранения.")

    scenarios: dict[str, MissionConfig] = st.session_state.scenarios
    if not scenarios:
        st.info("Сценарии ещё не сохранены.")
        return

    results = {slot: run_mission(cfg) for slot, cfg in scenarios.items()}
    order = [s for s in ("base", "modified") if s in results]

    def row(name: str, unit: str, fn) -> dict[str, object]:
        out: dict[str, object] = {"Показатель": name, "Единица": unit}
        for slot in order:
            out[SCENARIO_LABELS[slot]] = fn(results[slot])
        return out

    def status(m: MissionResult) -> str:
        if m.transfer.no_transfer:
            return "перелёт не нужен"
        return "выполним" if m.propellant.feasible else "невыполним"

    table = pd.DataFrame([
        row("Высота начальной орбиты", "км", lambda m: f"{m.config.initial_altitude_km:.1f}"),
        row("Высота целевой орбиты", "км", lambda m: f"{m.config.target_altitude_km:.1f}"),
        row("Сухая масса", "кг", lambda m: f"{m.config.dry_mass_kg:.1f}"),
        row("Запас топлива", "кг", lambda m: f"{m.config.propellant_kg:.1f}"),
        row("Удельный импульс", "с", lambda m: f"{m.config.isp_s:.1f}"),
        row("Суммарный Δv", "м/с", lambda m: f"{m.transfer.dv_total_mps:.1f}"),
        row("Время перелёта", "ч", lambda m: f"{m.transfer.t_transfer_s / 3600:.2f}"),
        row("Минимальное топливо", "кг", lambda m: f"{m.propellant.minimum_propellant_kg:.1f}"),
        row("Доступная Δv", "м/с", lambda m: f"{m.propellant.dv_available_mps:.1f}"),
        row("Бюджет топлива", "", status),
    ])
    st.dataframe(table, hide_index=True, width="stretch")

    fig = go.Figure(go.Bar(
        x=[SCENARIO_LABELS[s] for s in order],
        y=[results[s].propellant.minimum_propellant_kg for s in order],
        marker_color=[COLOR_MUTED if s == "base" else COLOR_ACCENT for s in order],
        text=[format_kg(results[s].propellant.minimum_propellant_kg) for s in order],
        textposition="outside", width=0.4,
    ))
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", font=dict(color=COLOR_TEXT, size=12),
        yaxis=dict(title="Минимальное топливо, кг", gridcolor=COLOR_BORDER, rangemode="tozero"),
        title=dict(text="Минимальная потребность в топливе", font=dict(size=14)),
    )
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})

    st.markdown(
        "- Изменение массы аппарата **не меняет** идеальную орбитальную Δv: она определяется "
        "только геометрией орбит.\n"
        "- Удельный импульс влияет на **потребность в топливе**, но не на геометрию перелёта, "
        "Δv и время.\n"
        "- Расход топлива (кг) и длительность перелёта (ч) — разные показатели; они не "
        "объединяются в общую оценку."
    )


def render_research() -> None:
    st.markdown("#### Исследовательский эксперимент")
    st.markdown(
        "**Гипотеза:** «Расчётная модель позволяет исследовать влияние высоты целевой орбиты и "
        "удельного импульса на параметры идеального орбитального перелёта.»"
    )
    st.caption(
        f"Эксперимент 1. Начальная орбита {SWEEP_INITIAL_ALTITUDE_KM:.0f} км; целевая высота "
        f"меняется от {SWEEP_TARGET_MIN_KM:.0f} до {SWEEP_TARGET_MAX_KM:,.0f} км "
        f"({SWEEP_POINTS} равномерных точек); сухая масса {SWEEP_DRY_MASS_KG:,.0f} кг; "
        f"Isp {SWEEP_ISP_S:.0f} с. Все значения рассчитаны моделью.".replace(",", " ")
    )
    df = _cached_altitude_sweep()
    metric = st.radio("Показатель", [COL_DV, COL_TIME, COL_MIN_PROP], horizontal=True,
                      key="sweep_metric")
    st.plotly_chart(_line_chart(df, COL_TARGET_ALT, metric), width="stretch",
                    config={"displaylogo": False})
    st.download_button("Скачать результаты эксперимента 1 (CSV)", dataframe_to_csv_bytes(df),
                       file_name="sweep_target_altitude.csv", mime="text/csv", on_click="ignore")

    with st.expander("Эксперимент 2: влияние удельного импульса"):
        c1, c2, c3 = st.columns(3)
        h1 = c1.number_input("Начальная орбита, км", ALTITUDE_MIN_KM, ALTITUDE_MAX_KM, 400.0,
                             step=50.0, key="isp_h1")
        h2 = c2.number_input("Целевая орбита, км", ALTITUDE_MIN_KM, ALTITUDE_MAX_KM, 35_786.0,
                             step=50.0, key="isp_h2")
        dry = c3.number_input("Сухая масса, кг", DRY_MASS_MIN_KG, DRY_MASS_MAX_KG,
                              SWEEP_DRY_MASS_KG, step=10.0, key="isp_dry")
        df2 = _cached_isp_sweep(float(h1), float(h2), float(dry))
        st.caption(
            f"Перелёт фиксирован (Δv = {format_dv(float(df2[COL_DV].iloc[0]))}); Isp меняется от "
            f"{ISP_SWEEP_MIN_S:.0f} до {ISP_SWEEP_MAX_S:.0f} с. Меняется только потребность "
            "в топливе — геометрия, Δv и время перелёта остаются прежними."
        )
        st.plotly_chart(_line_chart(df2, COL_ISP, COL_MIN_PROP), width="stretch",
                        config={"displaylogo": False})
        st.download_button("Скачать результаты эксперимента 2 (CSV)", dataframe_to_csv_bytes(df2),
                           file_name="sweep_isp.csv", mime="text/csv", on_click="ignore")


def tab_comparison() -> None:
    render_comparison()
    st.divider()
    render_research()


# ---------------------------------------------------------------------------
# Вкладка «О проекте»
# ---------------------------------------------------------------------------

def tab_about() -> None:
    st.markdown(f"""
#### Назначение
SpaceMission Lab — учебная расчётная лаборатория для изучения перелёта космического аппарата
между двумя круговыми компланарными орбитами Земли. {DISCLAIMER}

#### Для кого
Для школьников и студентов, изучающих программирование и космическую технику.

#### Перелёт Хомана
Аппарат выполняет два мгновенных касательных импульса. Первый переводит его с начальной
круговой орбиты на эллипс, касающийся обеих орбит; через половину периода эллипса второй
импульс скругляет орбиту на целевой высоте. При подъёме оба импульса разгоняют аппарат,
при снижении — тормозят. Перелёт выводит на нужную орбиту, но не обеспечивает встречу с
другим аппаратом.

#### Уравнение Циолковского
Связывает изменение скорости с отношением начальной и конечной масс:
Δv = Isp·g0·ln(m0/m_f). Удельный импульс характеризует экономичность двигателя, но не
определяет его тягу и длительность включения.

#### Константы
- μ = {MU_EARTH / 1e14:.9f}·10¹⁴ м³/с² — геоцентрическая гравитационная постоянная (WGS 84, IERS).
- R⊕ = {f"{R_EARTH:,.0f}".replace(",", " ")} м — опорный (средний) радиус Земли.
- g0 = {G0} м/с² — стандартное ускорение свободного падения.
""")

    st.dataframe(pd.DataFrame(EQUATIONS, columns=["Величина", "Формула"]), hide_index=True,
                 width="stretch")

    st.markdown(f"""
#### Технологии
Python 3.11+, Streamlit (интерфейс), NumPy (вычисления), Plotly (графики), pandas (таблицы),
pytest (автоматические тесты). Приложение работает локально, без сети и платных сервисов.

#### Практическая польза
- Наглядная связь формул орбитальной механики с геометрией орбит.
- Быстрая проверка, как высота орбиты и удельный импульс влияют на Δv и топливо.
- Воспроизводимые эксперименты с экспортом данных в CSV для дальнейшего анализа.

#### Методика исследования
1. Зафиксировать начальную орбиту (400 км), сухую массу и удельный импульс.
2. Изменять высоту целевой орбиты в документированном диапазоне.
3. Для каждой точки рассчитать Δv, время перелёта и минимальное топливо.
4. Отдельно изменять удельный импульс при фиксированном перелёте.
5. Сравнивать показатели только с одинаковыми единицами измерения.

Все данные получены расчётом; реальная телеметрия и лётные эксперименты не используются.

#### Ограничения
Модель не учитывает:
""")
    st.markdown("\n".join(f"- {item}" for item in LIMITATIONS))
    st.warning("Оперативное планирование космических миссий требует существенно более "
               "детальных моделей (несферичное поле, атмосфера, возмущения, конечные "
               "включения двигателя, навигация) и их независимой проверки.")
    st.caption(f"Версия модели: {MODEL_VERSION}")


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def main() -> None:
    _init_state()
    with st.container(horizontal_alignment="center"):
        with st.container(width=1100):
            st.title("SpaceMission Lab")
            st.markdown("Виртуальная лаборатория орбитальных перелётов")
            st.caption(DISCLAIMER)
            t1, t2, t3 = st.tabs(["Расчёт миссии", "Сравнение сценариев", "О проекте"])
            with t1:
                tab_mission()
            with t2:
                tab_comparison()
            with t3:
                tab_about()


main()
