import math

import numpy as np
import pytest

from orbital import MU_EARTH, R_EARTH, calculate_hohmann, specific_orbital_energy
from visualization import (
    KEPLER_MAX_ITER,
    build_orbit_figure,
    sample_transfer,
    solve_kepler,
    transfer_eccentricity,
    transfer_state,
)


class TestKepler:
    @pytest.mark.parametrize("e", [0.0, 0.1, 0.5, 0.75, 0.9, 0.99, 0.999])
    def test_residual_and_convergence(self, e):
        for m in np.linspace(0.0, 2 * math.pi, 73):
            sol = solve_kepler(float(m), e)
            assert sol.converged
            assert sol.iterations <= KEPLER_MAX_ITER
            residual = sol.eccentric_anomaly - e * math.sin(sol.eccentric_anomaly) - m
            assert abs(residual) < 1e-10

    def test_iteration_limit_reports_nonconvergence(self):
        sol = solve_kepler(1.0, 0.99, tol=0.0, max_iter=2)
        assert sol.iterations <= 2
        assert not sol.converged or sol.residual == 0.0

    @pytest.mark.parametrize("e", [-0.1, 1.0, 1.5, float("nan")])
    def test_invalid_eccentricity(self, e):
        with pytest.raises(ValueError):
            solve_kepler(1.0, e)

    def test_apsides(self):
        assert solve_kepler(0.0, 0.7).eccentric_anomaly == pytest.approx(0.0, abs=1e-12)
        assert solve_kepler(math.pi, 0.7).eccentric_anomaly == pytest.approx(math.pi, abs=1e-12)


class TestTrajectory:
    @pytest.mark.parametrize("h1,h2", [(400, 800), (400, 35_786), (1000, 400), (50_000, 200)])
    def test_endpoints(self, h1, h2):
        res = calculate_hohmann(h1, h2)
        start = transfer_state(res, 0.0)
        end = transfer_state(res, 1.0)
        assert start.x_m == pytest.approx(res.r1_m, rel=1e-9)
        assert start.y_m == pytest.approx(0.0, abs=1e-3)
        assert end.x_m == pytest.approx(-res.r2_m, rel=1e-9)
        assert end.y_m == pytest.approx(0.0, abs=1e-3)
        # Скорость в точках манёвров касательная и равна скорости на эллипсе
        assert start.vx_mps == pytest.approx(0.0, abs=1e-6)
        assert start.vy_mps == pytest.approx(res.vt1_mps, rel=1e-9)
        assert end.vy_mps == pytest.approx(-res.vt2_mps, rel=1e-9)
        assert end.elapsed_s == pytest.approx(res.t_transfer_s)

    def test_lowering_starts_at_apoapsis(self):
        res = calculate_hohmann(1000, 400)
        states = sample_transfer(res, 50)
        radii = [s.radius_m for s in states]
        assert radii[0] == pytest.approx(max(radii), rel=1e-9)
        assert radii[-1] == pytest.approx(min(radii), rel=1e-9)
        assert all(b <= a + 1e-6 for a, b in zip(radii, radii[1:]))

    def test_raising_radius_increases_monotonically(self):
        res = calculate_hohmann(400, 35_786)
        radii = [s.radius_m for s in sample_transfer(res, 100)]
        assert all(b >= a - 1e-6 for a, b in zip(radii, radii[1:]))

    @pytest.mark.parametrize("h1,h2", [(400, 35_786), (2000, 300)])
    def test_energy_and_angular_momentum_constant(self, h1, h2):
        res = calculate_hohmann(h1, h2)
        expected_energy = -MU_EARTH / (2 * res.a_transfer_m)
        h_ref = res.r1_m * res.vt1_mps
        for s in sample_transfer(res, 200):
            r = math.hypot(s.x_m, s.y_m)
            assert r == pytest.approx(s.radius_m, rel=1e-9)
            energy = specific_orbital_energy(r, math.hypot(s.vx_mps, s.vy_mps))
            assert energy == pytest.approx(expected_energy, rel=1e-9)
            assert s.x_m * s.vy_mps - s.y_m * s.vx_mps == pytest.approx(h_ref, rel=1e-9)

    def test_motion_is_not_uniform_in_angle(self):
        # При подъёме из перигея к середине времени пройдено больше 90° истинной аномалии
        res = calculate_hohmann(400, 35_786)
        mid = transfer_state(res, 0.5)
        assert mid.true_anomaly_rad > math.pi / 2 + 0.3

    def test_progress_clamped(self):
        res = calculate_hohmann(400, 800)
        assert transfer_state(res, 1.5).fraction == 1.0
        assert transfer_state(res, -0.2).fraction == 0.0
        with pytest.raises(ValueError):
            transfer_state(res, float("nan"))

    def test_no_transfer_state(self):
        res = calculate_hohmann(600, 600)
        assert transfer_eccentricity(res) == 0.0
        s = transfer_state(res, 0.7)
        assert s.x_m == pytest.approx(res.r1_m)
        assert s.elapsed_s == 0.0


class TestFigure:
    def test_figure_geometry(self):
        res = calculate_hohmann(400, 35_786)
        fig = build_orbit_figure(res, 0.3)
        assert fig.layout.yaxis.scaleanchor == "x"
        assert fig.layout.yaxis.scaleratio == 1
        names = [t.name for t in fig.data]
        assert any("Земля" in n for n in names)
        assert "Пройденная половина эллипса" in names
        assert "Точки манёвров" in names
        earth = next(t for t in fig.data if "Земля" in t.name)
        earth_r = max(math.hypot(x, y) for x, y in zip(earth.x, earth.y))
        assert earth_r == pytest.approx(R_EARTH / 1000, rel=1e-9)
        target = next(t for t in fig.data if t.name.startswith("Целевая"))
        target_r = max(math.hypot(x, y) for x, y in zip(target.x, target.y))
        assert target_r == pytest.approx(res.r2_m / 1000, rel=1e-9)

    def test_figure_no_transfer(self):
        fig = build_orbit_figure(calculate_hohmann(500, 500))
        names = [t.name for t in fig.data]
        assert "Пройденная половина эллипса" not in names
