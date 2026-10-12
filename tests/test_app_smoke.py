from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def _run():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    return at


def test_initial_render_has_tabs_and_no_results():
    at = _run()
    assert [t.label for t in at.tabs] == ["Расчёт миссии", "Сравнение сценариев", "О проекте"]
    assert at.title[0].value == "SpaceMission Lab"
    assert not at.metric


def test_calculate_then_invalidate_on_change():
    at = _run()
    _button(at, "Рассчитать перелёт").click().run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert labels == ["Суммарный Δv", "Время перелёта", "Минимальный запас топлива", "Запас Δv"]
    assert at.success

    at.number_input(key="in_h2").set_value(900.0).run()
    assert not at.exception
    assert not at.metric
    assert any("Параметры изменены" in i.value for i in at.info)


def test_infeasible_and_lowering_and_equal():
    at = _run()
    at.number_input(key="in_prop").set_value(0.0).run()
    _button(at, "Рассчитать перелёт").click().run()
    assert at.warning and "невыполним" in at.warning[0].value

    at.number_input(key="in_h1").set_value(1000.0)
    at.number_input(key="in_h2").set_value(400.0)
    at.number_input(key="in_prop").set_value(100.0).run()
    _button(at, "Рассчитать перелёт").click().run()
    assert not at.exception and at.success

    at.slider(key="progress").set_value(50).run()
    assert not at.exception

    at.number_input(key="in_h2").set_value(1000.0).run()
    _button(at, "Рассчитать перелёт").click().run()
    assert any("перелёт не требуется" in i.value for i in at.info)


def test_presets_and_scenario_comparison():
    at = _run()
    at.selectbox(key="preset_choice").set_value("Высокая орбита: 400 → 35 786 км").run()
    _button(at, "Применить пресет").click().run()
    assert at.number_input(key="in_h2").value == 35_786.0
    _button(at, "Рассчитать перелёт").click().run()
    _button(at, "Сохранить текущий расчёт как «Базовый сценарий»").click().run()

    at.number_input(key="in_isp").set_value(450.0).run()
    _button(at, "Рассчитать перелёт").click().run()
    _button(at, "Сохранить текущий расчёт как «Изменённый сценарий»").click().run()
    assert not at.exception
    assert set(at.session_state.scenarios) == {"base", "modified"}

    at.radio(key="sweep_metric").set_value("Время перелёта, ч").run()
    assert not at.exception
