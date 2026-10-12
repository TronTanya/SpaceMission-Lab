import math

import pytest

from orbital import (
    MU_EARTH,
    R_EARTH,
    OrbitalError,
    altitude_km_to_radius_m,
    calculate_hohmann,
    circular_velocity,
    orbital_period,
    specific_orbital_energy,
    validate_altitude_km,
    vis_viva,
)


class TestValidationAndUnits:
    def test_altitude_converted_to_geocentric_radius_in_metres(self):
        assert altitude_km_to_radius_m(400) == pytest.approx(6_771_000.0)
        assert altitude_km_to_radius_m(35_786) == pytest.approx(R_EARTH + 35_786_000.0)

    @pytest.mark.parametrize("bad", [199.9, 50_000.1, -1, 0, float("nan"), float("inf"), -float("inf")])
    def test_rejects_out_of_range_and_nonfinite(self, bad):
        with pytest.raises(OrbitalError):
            validate_altitude_km(bad)

    @pytest.mark.parametrize("bad", ["400", None, True, [400]])
    def test_rejects_non_numeric(self, bad):
        with pytest.raises(OrbitalError):
            validate_altitude_km(bad)

    def test_range_boundaries_accepted(self):
        assert validate_altitude_km(200) == 200.0
        assert validate_altitude_km(50_000) == 50_000.0

    def test_hohmann_validates_inputs(self):
        with pytest.raises(OrbitalError):
            calculate_hohmann(100, 800)
        with pytest.raises(OrbitalError):
            calculate_hohmann(400, float("nan"))


class TestCircularOrbit:
    def test_leo_velocity_and_period_reference(self):
        # r = 6771 km: v ≈ 7.6726 км/с, T ≈ 92.4 мин (ручной расчёт √(μ/r), 2π√(r³/μ))
        r = 6_771_000.0
        assert circular_velocity(r) == pytest.approx(7672.6, abs=0.5)
        assert orbital_period(r) / 60 == pytest.approx(92.41, abs=0.05)

    def test_geostationary_radius_gives_sidereal_day(self):
        # r = 42 164 км — радиус ГСО; период ≈ звёздные сутки 86 164 с
        assert orbital_period(42_164_000.0) == pytest.approx(86_164.0, abs=5.0)
        assert circular_velocity(42_164_000.0) == pytest.approx(3074.7, abs=0.5)

    def test_velocity_and_period_consistent(self):
        r = 10_000_000.0
        assert orbital_period(r) == pytest.approx(2 * math.pi * r / circular_velocity(r), rel=1e-12)

    def test_invalid_radius(self):
        with pytest.raises(OrbitalError):
            circular_velocity(0)
        with pytest.raises(OrbitalError):
            orbital_period(-1)


class TestHohmann:
    def test_equal_orbits_need_no_transfer(self):
        res = calculate_hohmann(500, 500)
        assert res.no_transfer
        assert res.dv1_signed_mps == 0.0
        assert res.dv2_signed_mps == 0.0
        assert res.dv_total_mps == 0.0
        assert res.t_transfer_s == 0.0
        assert res.direction1 == "не требуется"
        assert res.period1_s == pytest.approx(res.period2_s)

    def test_raising_has_positive_impulses(self):
        res = calculate_hohmann(400, 800)
        assert res.is_raising and not res.is_lowering
        assert res.dv1_signed_mps > 0
        assert res.dv2_signed_mps > 0
        assert "ускорение" in res.direction1 and "ускорение" in res.direction2

    def test_lowering_has_negative_impulses(self):
        res = calculate_hohmann(1000, 400)
        assert res.is_lowering and not res.is_raising
        assert res.dv1_signed_mps < 0
        assert res.dv2_signed_mps < 0
        assert "торможение" in res.direction1 and "торможение" in res.direction2

    @pytest.mark.parametrize("h1,h2", [(400, 800), (300, 35_786), (200, 50_000)])
    def test_raise_lower_symmetry(self, h1, h2):
        up = calculate_hohmann(h1, h2)
        down = calculate_hohmann(h2, h1)
        assert up.dv_total_mps == pytest.approx(down.dv_total_mps, rel=1e-12)
        assert up.t_transfer_s == pytest.approx(down.t_transfer_s, rel=1e-12)
        assert up.dv1_abs_mps == pytest.approx(down.dv2_abs_mps, rel=1e-12)
        assert up.dv2_abs_mps == pytest.approx(down.dv1_abs_mps, rel=1e-12)

    def test_total_is_sum_of_magnitudes(self):
        res = calculate_hohmann(400, 800)
        assert res.dv_total_mps == pytest.approx(abs(res.dv1_signed_mps) + abs(res.dv2_signed_mps))

    def test_transfer_time_is_half_ellipse_period(self):
        res = calculate_hohmann(400, 35_786)
        a = (res.r1_m + res.r2_m) / 2
        assert res.a_transfer_m == pytest.approx(a)
        assert res.t_transfer_s == pytest.approx(math.pi * math.sqrt(a ** 3 / MU_EARTH), rel=1e-12)
        assert res.t_transfer_s == pytest.approx(0.5 * orbital_period(a), rel=1e-12)
        # Известный порядок величины: перелёт НОО → ГСО длится около 5,3 ч, Δv ≈ 3,85–3,9 км/с
        assert 5.0 < res.t_transfer_s / 3600 < 5.5
        assert 3800 < res.dv_total_mps < 3950

    def test_vis_viva_consistency_at_endpoints(self):
        res = calculate_hohmann(400, 20_000)
        a = res.a_transfer_m
        assert res.vt1_mps == pytest.approx(vis_viva(res.r1_m, a), rel=1e-12)
        assert res.vt2_mps == pytest.approx(vis_viva(res.r2_m, a), rel=1e-12)
        e1 = specific_orbital_energy(res.r1_m, res.vt1_mps)
        e2 = specific_orbital_energy(res.r2_m, res.vt2_mps)
        assert e1 == pytest.approx(-MU_EARTH / (2 * a), rel=1e-12)
        assert e2 == pytest.approx(-MU_EARTH / (2 * a), rel=1e-12)
        # Сохранение момента импульса в апсидах: r1·vt1 = r2·vt2
        assert res.r1_m * res.vt1_mps == pytest.approx(res.r2_m * res.vt2_mps, rel=1e-12)

    def test_mass_independent(self):
        # Орбитальная Δv не зависит от массы аппарата — функция массу не принимает вовсе
        a = calculate_hohmann(400, 800)
        b = calculate_hohmann(400, 800)
        assert a == b
