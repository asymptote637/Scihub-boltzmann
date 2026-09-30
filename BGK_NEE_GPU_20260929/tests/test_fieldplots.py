import numpy as np
import pytest
from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from dashboard.fieldplots import REGIONS, draw_field, region_bounds, streamline_grid


@pytest.fixture(autouse=True)
def fonts():
    with rc_context({"font.family": ["Microsoft YaHei", "DejaVu Sans"]}):
        yield


def field(x=None, y=None):
    x = np.linspace(0, 1, 32) if x is None else x
    y = np.linspace(0, 1, 32) if y is None else y
    xx, yy = np.meshgrid(x, y)
    return dict(x=x, y=y, ux=.04 * (yy - .5), uy=-.04 * (xx - .5), rho=np.ones_like(xx))


@pytest.mark.parametrize("region", REGIONS)
def test_regions_and_interpolation_preserve_coordinates(region):
    data = field(np.array([0., .12, .41, .8, .96, 1.]), np.array([0., .22, .5, .7, .99, 1.]))
    original = {key: value.copy() for key, value in data.items()}
    bounds = region_bounds(data, region)
    x, y, u, v = streamline_grid(data, bounds)
    assert np.allclose(u, .04 * (y[:, None] - .5))
    assert np.allclose(v, -.04 * (x[None, :] - .5))
    assert np.allclose(np.diff(x), np.diff(x)[0])
    assert np.allclose(np.diff(y), np.diff(y)[0])
    assert (x[0], x[-1], y[0], y[-1]) == bounds
    for key in original:
        np.testing.assert_array_equal(original[key], data[key])


def test_bounded_grid_and_corner_native_resolution():
    data = field(np.linspace(0, 1, 1024), np.linspace(0, 1, 1024))
    assert streamline_grid(data, region_bounds(data))[2].shape == (257, 257)
    assert streamline_grid(data, region_bounds(data, "bottom_left", .1))[2].shape == (103, 103)


@pytest.mark.parametrize("mode", ["speed", "ux", "uy", "rho", "streamlines", "overlay", "vortices"])
def test_plot_modes_and_zero_field(mode):
    data = field()
    fig = Figure(figsize=(8, 7))
    canvas = FigureCanvasAgg(fig)
    axes = draw_field(fig, data, mode, speed=.04, density=.6, region="bottom_right", caption="Re 100 - step 301")
    assert len(axes) == (4 if mode == "vortices" else 1)
    if mode != "vortices":
        assert axes[0].get_xlim() == pytest.approx((.65, 1))
    canvas.draw()
    assert np.asarray(canvas.buffer_rgba()).std() > 5
    data["ux"][:] = data["uy"][:] = 0
    draw_field(fig, data, mode, speed=.04, density=.6)
    canvas.draw()


def test_nonfinite_streamlines_are_rejected():
    data = field()
    data["ux"][1, 2] = np.nan
    with pytest.raises(ValueError, match="非有限"):
        streamline_grid(data, region_bounds(data))


@pytest.mark.parametrize("options", [dict(speed=0), dict(speed=np.nan), dict(speed=.04, density=0),
                                      dict(speed=.04, fraction=.9), dict(speed=.04, region="missing")])
def test_invalid_plot_options(options):
    with pytest.raises(ValueError):
        draw_field(Figure(), field(), "streamlines", **options)
