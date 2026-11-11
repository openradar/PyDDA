"""
========================================
pydda.terrain (pydda.terrain)
========================================

.. currentmodule:: pydda.terrain

Terrain handling for wind retrievals in complex terrain.

PyDDA's default lower boundary condition is flat impermeability: ``w = 0`` on
the bottom plane of the analysis domain. Over terrain the correct statement is
no-penetration through the terrain surface,

.. math::

    \\vec{u} \\cdot \\hat{n} = 0, \\qquad
    \\hat{n} = \\frac{(-h_x, -h_y, 1)}{\\sqrt{1 + h_x^2 + h_y^2}},

which is what makes air climb a ridge. This module builds the geometry needed
to impose that condition with a ghost-cell immersed boundary method (IBM) from
a digital elevation model, for use with the ``Cterrain`` term of
:py:func:`pydda.retrieval.get_dd_wind_field`.

Vertical datum
--------------

Analysis heights in PyDDA are **orthometric** (above mean sea level):
``point_altitude = origin_altitude + point_z``, and ``origin_altitude`` comes
from the radar metadata, which for NEXRAD, ARM and NASA ground radars is
antenna height above MSL.

The DEM must therefore also be orthometric. SRTM v3 (EGM96), ASTER GDEM
(EGM96), Copernicus GLO-30 (EGM2008) and USGS 3DEP (NAVD88) are all safe at
PyDDA's vertical grid spacing. An **ellipsoidal** DEM (WGS84/ITRF heights, as
in some TanDEM-X and ICESat-2 products) is not: the geoid-ellipsoid separation
runs from about -107 m to +85 m globally, and applies as a systematic offset to
every column. Convert to orthometric heights first.

:py:func:`check_datum` turns this into a diagnostic rather than an assumption.

.. autosummary::
    :toctree: generated/

    check_datum
    terrain_ghost_cells

"""

import numpy as np
import pyart

from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter


def check_datum(Grid, dem, tower_max=50.0, verbose=True):
    """
    Verify that a DEM and the radar altitudes share a vertical datum.

    Samples the DEM at each radar position in *Grid* and compares it with the
    reported radar altitude. A consistent pair leaves a small **positive**
    residual, which is the antenna tower height. A large residual of either
    sign means the DEM is on a different vertical datum -- most often
    ellipsoidal rather than orthometric -- and must be converted before it is
    used to build a terrain boundary condition.

    Parameters
    ----------
    Grid: xarray.Dataset
        A PyDDA grid, as returned by :func:`pydda.io.read_grid`.
    dem: xarray.DataArray
        Elevation in m, with 1D "lat" and "lon" coordinates.
    tower_max: float
        Residuals larger than this in magnitude, in m, are reported as a
        likely datum mismatch.
    verbose: bool
        Print a line per radar.

    Returns
    -------
    residuals: 1D float array
        ``radar_altitude - dem_elevation`` at each radar, in m.
    """
    interp = _dem_interpolator(dem)
    lat = np.atleast_1d(np.asarray(Grid["radar_latitude"].values).ravel())
    lon = np.atleast_1d(np.asarray(Grid["radar_longitude"].values).ravel())
    alt = np.atleast_1d(np.asarray(Grid["radar_altitude"].values).ravel())

    resid = alt - interp(np.stack([lat, lon], axis=-1))

    if verbose:
        for n, r in enumerate(resid):
            if np.abs(r) > tower_max:
                print(
                    "  WARNING radar %d: radar_altitude - DEM = %+.1f m. A small "
                    "positive tower height was expected. Check the DEM's vertical "
                    "datum; see the notes in pydda.terrain." % (n, r)
                )
            else:
                print("  radar %d: tower height %+.1f m, datum consistent." % (n, r))
    return resid


def terrain_ghost_cells(
    Grid, dem, refine=4, smooth_cells=1.0, image_dist=1.75, verbose=True
):
    """
    Build the ghost-cell immersed-boundary structure for an analysis grid.

    Each solid cell adjacent to the fluid is reflected across the terrain
    surface to an image point, where the velocity is interpolated trilinearly
    from the surrounding fluid cells. The two are related by

    .. math::

        \\vec{u}_G = M_G \\vec{u}_I, \\qquad
        M_G = I - (1 + \\beta_G)\\hat{n}\\hat{n}^T, \\qquad
        \\beta_G = d_G / d_I

    with :math:`d_G` the distance from the ghost cell down to the surface and
    :math:`d_I` the distance from the surface up to the image point. The normal
    velocity varies linearly across the wall and so vanishes on it; the
    tangential components have zero normal derivative. When :math:`d_I = d_G`
    this is the Householder reflection :math:`I - 2\\hat{n}\\hat{n}^T`; larger
    :math:`d_I` lets the image point stand off far enough for a
    well-conditioned interpolation stencil.

    The resulting relation is *linear* in the wind field, so it enters the
    retrieval as a quadratic penalty via the ``Cterrain`` weight of
    :py:func:`pydda.retrieval.get_dd_wind_field`.

    Parameters
    ----------
    Grid: xarray.Dataset
        A PyDDA grid, as returned by :func:`pydda.io.read_grid`.
    dem: xarray.DataArray
        Orthometric (above-MSL) elevation in m, with 1D "lat" and "lon"
        coordinates covering the grid domain. Run :func:`check_datum` first.
    refine: int
        DEM sub-samples per analysis cell along each horizontal axis. The
        samples are block-averaged to the grid scale, so a DEM much finer than
        the analysis grid is used properly rather than point-sampled.
    smooth_cells: float
        Gaussian smoothing of the grid-scale terrain, in analysis cells,
        applied before the slopes are computed. Raw 30 m DEM slopes are far
        steeper than the resolved flow can respond to, and differencing them
        injects vertical velocity that no radar observed.
    image_dist: float
        Minimum image-point standoff from the terrain surface, in units of the
        characteristic cell size ``(dx dy dz)**(1/3)``.
    verbose: bool
        Print a summary of the resulting geometry.

    Returns
    -------
    terrain: dict
        Dictionary with the following entries, suitable for passing to the
        ``terrain`` keyword of :py:func:`pydda.retrieval.get_dd_wind_field`:

        h: 2D float array
            Grid-scale terrain elevation in m above MSL.
        fluid: 3D bool array
            True where the cell centre is above the terrain surface.
        phi: 3D float array
            Signed distance to the terrain surface in m, positive in the fluid.
        ghost_flat: 1D int array
            Flat indices of the ghost cells.
        stencil_flat: 2D int array
            (n_ghost, 8) flat indices of each image point's stencil.
        stencil_w: 2D float array
            (n_ghost, 8) trilinear weights, rows summing to one.
        M: 3D float array
            (n_ghost, 3, 3) wall operators.
        solid_free_flat: 1D int array
            Flat indices of solid cells that are not ghost cells. These carry
            no boundary information and are damped towards zero so that they
            do not wander during the optimization.
        n_dropped: int
            Ghost cells discarded because no fluid cell was available to
            interpolate from. A large count means the grid is too coarse for
            the terrain.
    """
    x = Grid["x"].values
    y = Grid["y"].values
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    nx, ny = len(x), len(y)

    origin_alt = float(np.asarray(Grid["origin_altitude"].values).ravel()[0])
    z_msl = Grid["z"].values + origin_alt
    dz = float(z_msl[1] - z_msl[0])
    nz = len(z_msl)
    delta = (dx * dy * dz) ** (1.0 / 3.0)

    # Terrain height on the analysis columns, and its slopes. The slopes come
    # from the smoothed field; see the smooth_cells docstring.
    h = _sample_dem(Grid, dem, refine, x, y, dx, dy, nx, ny)
    h = gaussian_filter(h, smooth_cells, mode="nearest")
    hy, hx = np.gradient(h, dy, dx)

    # Signed distance to a height field. Exact for planar terrain; see the
    # limitations note in the module documentation of pydda.cost_functions.
    slope_norm = np.sqrt(1.0 + hx**2 + hy**2)
    phi = (z_msl[:, None, None] - h[None, ...]) / slope_norm[None, ...]
    fluid = phi > 0.0
    solid = ~fluid

    # A ghost cell is a solid cell with at least one face neighbour in the fluid
    nbr = np.zeros_like(fluid)
    for ax in (0, 1, 2):
        for shift in (-1, 1):
            nbr |= np.roll(fluid, shift, axis=ax)
    ghost = solid & nbr
    gk, gj, gi = np.nonzero(ghost)

    nvec = np.stack(
        [
            -hx[gj, gi] / slope_norm[gj, gi],
            -hy[gj, gi] / slope_norm[gj, gi],
            1.0 / slope_norm[gj, gi],
        ],
        axis=1,
    )
    d_G = -phi[gk, gj, gi]
    d_I = np.maximum(d_G, image_dist * delta)
    beta = d_G / d_I

    x_ghost = np.stack([x[gi], y[gj], z_msl[gk]], axis=1)
    x_image = x_ghost + (d_G + d_I)[:, None] * nvec

    stencil_flat, stencil_w = _trilinear_stencil(
        x_image, x, y, z_msl, dx, dy, dz, nx, ny, nz
    )

    # Drop non-fluid corners from the stencil and renormalize. Ghost cells left
    # with no fluid corner at all are removed from the constraint set entirely,
    # and counted rather than silently ignored.
    stencil_w = np.where(fluid.ravel()[stencil_flat], stencil_w, 0.0)
    wsum = stencil_w.sum(axis=1)
    keep = wsum > 1e-8
    n_dropped = int((~keep).sum())

    stencil_w = stencil_w[keep] / wsum[keep, None]
    stencil_flat = stencil_flat[keep]
    nvec = nvec[keep]
    beta = beta[keep]
    ghost_flat = (gk * ny * nx + gj * nx + gi)[keep]

    M = (
        np.eye(3)[None, ...]
        - (1.0 + beta)[:, None, None] * nvec[:, :, None] * nvec[:, None, :]
    )

    # Solid cells that carry no boundary information
    solid_flat = np.nonzero(solid.ravel())[0]
    solid_free_flat = np.setdiff1d(solid_flat, ghost_flat, assume_unique=False)

    if verbose:
        print(
            "Terrain: %d ghost cells, %d dropped, %d damped solid cells, "
            "%.1f%% of the domain is fluid."
            % (
                len(ghost_flat),
                n_dropped,
                len(solid_free_flat),
                100.0 * fluid.mean(),
            )
        )
        if n_dropped > 0.05 * max(len(ghost_flat), 1):
            print(
                "  WARNING: more than 5%% of ghost cells lacked a fluid stencil. "
                "The analysis grid may be too coarse to resolve this terrain."
            )

    return {
        "h": h,
        "fluid": fluid,
        "phi": phi,
        "ghost_flat": ghost_flat,
        "stencil_flat": stencil_flat,
        "stencil_w": stencil_w,
        "M": M,
        "solid_free_flat": solid_free_flat,
        "n_dropped": n_dropped,
    }


def _dem_interpolator(dem):
    """Build a RegularGridInterpolator over a DEM with ascending lat/lon."""
    lat = dem["lat"].values
    lon = dem["lon"].values
    if lat[1] < lat[0]:
        lat = lat[::-1]
        dem = dem.isel(lat=slice(None, None, -1))
    if lon[1] < lon[0]:
        lon = lon[::-1]
        dem = dem.isel(lon=slice(None, None, -1))
    return RegularGridInterpolator(
        (lat, lon),
        np.asarray(dem.values, dtype=np.float64),
        bounds_error=False,
        fill_value=None,
    )


def _sample_dem(Grid, dem, refine, x, y, dx, dy, nx, ny):
    """Block-average a DEM onto the analysis columns."""
    off = (np.arange(refine) + 0.5) / refine - 0.5
    xf = (x[:, None] + off[None, :] * dx).ravel()
    yf = (y[:, None] + off[None, :] * dy).ravel()
    Xf, Yf = np.meshgrid(xf, yf)

    projparams = {
        "proj": "pyart_aeqd",
        "lon_0": float(np.asarray(Grid["origin_longitude"].values).ravel()[0]),
        "lat_0": float(np.asarray(Grid["origin_latitude"].values).ravel()[0]),
    }
    lon, lat = pyart.core.cartesian_to_geographic(Xf, Yf, projparams)

    h_fine = _dem_interpolator(dem)(np.stack([lat, lon], axis=-1))
    return h_fine.reshape(ny, refine, nx, refine).mean(axis=(1, 3))


def _trilinear_stencil(pts, x, y, z, dx, dy, dz, nx, ny, nz):
    """Flat indices and trilinear weights of the 8 cells around each point."""
    fi = np.clip((pts[:, 0] - x[0]) / dx, 0, nx - 1.001)
    fj = np.clip((pts[:, 1] - y[0]) / dy, 0, ny - 1.001)
    fk = np.clip((pts[:, 2] - z[0]) / dz, 0, nz - 1.001)
    i0 = fi.astype(np.int64)
    j0 = fj.astype(np.int64)
    k0 = fk.astype(np.int64)
    ti, tj, tk = fi - i0, fj - j0, fk - k0

    idx = np.empty((len(pts), 8), dtype=np.int64)
    wts = np.empty((len(pts), 8), dtype=np.float64)
    for c, (a, b, d) in enumerate(np.ndindex(2, 2, 2)):
        idx[:, c] = (k0 + d) * ny * nx + (j0 + b) * nx + (i0 + a)
        wts[:, c] = (
            (ti if a else 1.0 - ti) * (tj if b else 1.0 - tj) * (tk if d else 1.0 - tk)
        )
    return idx, wts
