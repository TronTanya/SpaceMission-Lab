import math

import pytest

from orbital import G0
from propulsion import (
    PropulsionError,
    available_delta_v,
    calculate_propellant,
    exhaust_velocity,
    minimum_propellant,
    validate_dry_mass,
    validate_isp,
    validate_propellant,
)


class TestValidation:
    @pytest.mark.parametrize("bad", [0.5, 100_001, float("nan"), float("inf"), "10", None, True])
    def test_dry_mass(self, bad):
        with pytest.raises(PropulsionError):
            validate_dry_mass(bad)

    @pytest.mark.parametrize("bad", [-0.1, 100_001, float("nan"), float("-inf")])
    def test_propellant(self, bad):
        with pytest.raises(PropulsionError):
            validate_propellant(bad)

    @pytest.mark.parametrize("bad", [49.9, 5000.1, float("nan"), 0])
    def test_isp(self, bad):
        with pytest.raises(PropulsionError):
            validate_isp(bad)

    def test_negative_or_nonfinite_dv_rejected(self):
        with pytest.raises(PropulsionError):
            calculate_propellant(100, 10, 300, -1)
        with pytest.raises(PropulsionError):
            calculate_propellant(100, 10, 300, float("inf"))

    def test_exhaust_velocity_uses_g0(self):
        assert exhaust_velocity(300) == pytest.approx(300 * 9.80665)
        assert G0 == 9.80665


class TestRocketEquation:
    def test_available_dv_mass_ratio_e(self):
        # m0 / m_dry = e  →  Δv = Isp · g0
        dry = 1000.0
        prop = dry * (math.e - 1)
        assert available_delta_v(dry, prop, 300) == pytest.approx(300 * G0, rel=1e-12)

    def test_zero_propellant_gives_zero_dv(self):
        assert available_delta_v(500, 0, 300) == 0.0

    def test_zero_propellant_infeasible_for_nonzero_dv(self):
        res = calculate_propellant(500, 0, 300, 100)
        assert not res.feasible
        assert res.remaining_propellant_kg is None
        assert res.dv_margin_mps is None
        assert res.dv_shortfall_mps == pytest.approx(100)

    def test_zero_propellant_zero_dv_is_feasible(self):
        res = calculate_propellant(500, 0, 300, 0)
        assert res.feasible
        assert res.remaining_propellant_kg == 0.0
        assert res.minimum_propellant_kg == 0.0
        assert res.propellant_required_kg == 0.0

    def test_minimum_propellant_formula(self):
        dry, dv, isp = 1000.0, 3000.0, 320.0
        expected = dry * (math.exp(dv / (isp * G0)) - 1)
        assert minimum_propellant(dry, dv, isp) == pytest.approx(expected, rel=1e-12)

    def test_minimum_propellant_exactly_enough(self):
        dry, dv, isp = 800.0, 1500.0, 310.0
        mp = minimum_propellant(dry, dv, isp)
        assert available_delta_v(dry, mp, isp) == pytest.approx(dv, rel=1e-12)
        res = calculate_propellant(dry, mp, isp, dv)
        assert res.feasible
        assert res.remaining_propellant_kg == pytest.approx(0.0, abs=1e-6)
        assert res.dv_margin_mps == pytest.approx(0.0, abs=1e-6)

    def test_feasible_budget_remaining_and_margin(self):
        dry, prop, isp, dv = 500.0, 150.0, 300.0, 214.0
        res = calculate_propellant(dry, prop, isp, dv)
        m0 = dry + prop
        mf = m0 * math.exp(-dv / (isp * G0))
        assert res.feasible
        assert res.initial_mass_kg == pytest.approx(m0)
        assert res.final_mass_kg == pytest.approx(mf, rel=1e-12)
        assert res.propellant_required_kg == pytest.approx(m0 - mf, rel=1e-12)
        assert res.remaining_propellant_kg == pytest.approx(prop - (m0 - mf), rel=1e-12)
        assert res.remaining_propellant_kg > 0
        assert res.dv_margin_mps == pytest.approx(res.dv_available_mps - dv, rel=1e-12)
        # Расход с полной заправкой больше минимума для сухого аппарата
        assert res.propellant_required_kg > res.minimum_propellant_kg

    def test_infeasible_budget(self):
        res = calculate_propellant(1500, 100, 300, 3900)
        assert not res.feasible
        assert res.final_mass_kg < res.dry_mass_kg
        assert res.remaining_propellant_kg is None
        assert res.dv_margin_mps is None
        assert res.dv_shortfall_mps == pytest.approx(3900 - res.dv_available_mps)
        assert res.dv_shortfall_mps > 0
        assert res.minimum_propellant_kg > res.propellant_available_kg

    def test_higher_isp_needs_less_propellant(self):
        assert minimum_propellant(1000, 3000, 450) < minimum_propellant(1000, 3000, 300)
