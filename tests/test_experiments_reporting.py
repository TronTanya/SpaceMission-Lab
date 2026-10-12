import json
import math

import pytest

from experiments import (
    COL_DV,
    COL_ISP,
    COL_MIN_PROP,
    COL_TARGET_ALT,
    COL_TIME,
    PRESETS,
    MissionConfig,
    altitude_sweep,
    isp_sweep,
    run_mission,
)
from orbital import calculate_hohmann
from propulsion import minimum_propellant
from reporting import (
    ConfigImportError,
    build_html_report,
    config_to_json,
    dataframe_to_csv_bytes,
    parse_config_json,
    results_dataframe,
)


class TestSweeps:
    def test_altitude_sweep_matches_engine(self):
        df = altitude_sweep(points=25)
        assert len(df) == 25
        assert list(df.columns) == [COL_TARGET_ALT, COL_DV, COL_TIME, COL_MIN_PROP]
        assert df[COL_TARGET_ALT].iloc[0] == pytest.approx(500)
        assert df[COL_TARGET_ALT].iloc[-1] == pytest.approx(50_000)
        for _, row in df.sample(5, random_state=0).iterrows():
            ref = calculate_hohmann(400, row[COL_TARGET_ALT])
            assert row[COL_DV] == pytest.approx(ref.dv_total_mps, rel=1e-12)
            assert row[COL_TIME] == pytest.approx(ref.t_transfer_s / 3600, rel=1e-12)
            assert row[COL_MIN_PROP] == pytest.approx(
                minimum_propellant(1000, ref.dv_total_mps, 320), rel=1e-12)
        assert df[COL_TIME].is_monotonic_increasing
        assert df[COL_MIN_PROP].is_monotonic_increasing == df[COL_DV].is_monotonic_increasing

    def test_altitude_sweep_reproducible(self):
        assert altitude_sweep().equals(altitude_sweep())

    def test_isp_sweep_fixed_transfer(self):
        df = isp_sweep(points=10)
        assert list(df.columns) == [COL_ISP, COL_DV, COL_MIN_PROP]
        assert df[COL_DV].nunique() == 1
        assert df[COL_MIN_PROP].is_monotonic_decreasing

    def test_sweep_rejects_bad_ranges(self):
        with pytest.raises(ValueError):
            altitude_sweep(target_min_km=1000, target_max_km=500)
        with pytest.raises(ValueError):
            altitude_sweep(points=1)


class TestPresets:
    def test_presets_are_inputs_and_run(self):
        for name, cfg in PRESETS.items():
            assert isinstance(cfg, MissionConfig)
            m = run_mission(cfg)
            assert m.transfer.dv_total_mps > 0, name
        lowering = run_mission(PRESETS["Снижение орбиты: 1 000 → 400 км"])
        assert lowering.transfer.is_lowering


VALID = {
    "initial_altitude_km": 400,
    "target_altitude_km": 800.5,
    "dry_mass_kg": 500,
    "propellant_kg": 150,
    "isp_s": 300,
}


def _payload(**overrides):
    payload = {"app": "SpaceMission Lab", "schema_version": 1, "config": dict(VALID)}
    payload.update(overrides)
    return payload


class TestJson:
    def test_roundtrip(self):
        cfg = MissionConfig(400, 35_786, 1500, 4000, 320)
        assert parse_config_json(config_to_json(cfg)) == cfg
        assert parse_config_json(config_to_json(cfg).encode("utf-8-sig")) == cfg

    @pytest.mark.parametrize("text", ["", "not json", "[1,2]", "42", '"str"'])
    def test_rejects_malformed(self, text):
        with pytest.raises(ConfigImportError):
            parse_config_json(text)

    @pytest.mark.parametrize("version", [2, 0, "1", 1.0, True, None])
    def test_rejects_unsupported_version(self, version):
        with pytest.raises(ConfigImportError):
            parse_config_json(json.dumps(_payload(schema_version=version)))

    def test_rejects_missing_version(self):
        payload = _payload()
        del payload["schema_version"]
        with pytest.raises(ConfigImportError):
            parse_config_json(json.dumps(payload))

    @pytest.mark.parametrize("value", ["400", None, True, [400], {"v": 1}])
    def test_rejects_invalid_types(self, value):
        payload = _payload()
        payload["config"]["initial_altitude_km"] = value
        with pytest.raises(ConfigImportError):
            parse_config_json(json.dumps(payload))

    @pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1e400"])
    def test_rejects_nonfinite(self, literal):
        text = json.dumps(_payload()).replace("150", literal)
        with pytest.raises(ConfigImportError):
            parse_config_json(text)

    def test_rejects_out_of_range(self):
        payload = _payload()
        payload["config"]["isp_s"] = 10
        with pytest.raises(ConfigImportError):
            parse_config_json(json.dumps(payload))

    def test_rejects_missing_and_unknown_fields(self):
        payload = _payload()
        del payload["config"]["isp_s"]
        with pytest.raises(ConfigImportError):
            parse_config_json(json.dumps(payload))
        payload = _payload()
        payload["config"]["__import__"] = 1
        with pytest.raises(ConfigImportError):
            parse_config_json(json.dumps(payload))

    def test_rejects_non_object_config(self):
        with pytest.raises(ConfigImportError):
            parse_config_json(json.dumps(_payload(config=[1, 2, 3])))

    def test_rejects_huge_file(self):
        with pytest.raises(ConfigImportError):
            parse_config_json(b" " * (70 * 1024))


class TestExport:
    def test_csv_has_bom_and_russian_labels(self):
        mission = run_mission(MissionConfig(400, 800, 500, 150, 300))
        data = dataframe_to_csv_bytes(results_dataframe(mission))
        assert data.startswith(b"\xef\xbb\xbf")
        text = data.decode("utf-8-sig")
        assert "Суммарный Δv" in text
        assert "Выполним по топливу" in text

    def test_infeasible_csv_has_no_remaining_propellant(self):
        mission = run_mission(MissionConfig(400, 35_786, 1500, 100, 300))
        text = dataframe_to_csv_bytes(results_dataframe(mission)).decode("utf-8-sig")
        assert "Невыполним" in text
        assert "Остаток топлива" not in text
        assert "Теоретическая потребность" in text

    def test_html_report_complete_and_escaped(self):
        mission = run_mission(MissionConfig(400, 800, 500, 150, 300))
        report = build_html_report(mission)
        assert report.startswith("<!DOCTYPE html>")
        for needle in ("Константы", "Уравнения", "Ограничения модели", "версия модели",
                       "3.986004418", "Статус бюджета топлива"):
            assert needle in report
        assert "<script" not in report.lower()

    def test_html_table_escapes_content(self):
        from reporting import _table

        out = _table([("<script>alert(1)</script>", "a & b")], ("<h>",))
        assert "<script>" not in out
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in out
        assert "a &amp; b" in out
        assert "&lt;h&gt;" in out

    def test_no_transfer_report(self):
        mission = run_mission(MissionConfig(500, 500, 500, 0, 300))
        assert mission.propellant.feasible
        assert "Перелёт не требуется" in build_html_report(mission)
        assert math.isclose(mission.propellant.minimum_propellant_kg, 0.0)
