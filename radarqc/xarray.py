from typing import List
import warnings

import numpy as np

try:
    import xarray as xr
except ModuleNotFoundError:
    xr = None
    warnings.warn(
        "Could not find package 'xarray', continuing with compatibility"
        "functionality disabled",
        ImportWarning,
    )

from radarqc.header import CSFileHeader
from radarqc.spectrum import Spectrum


_FREQUENCY_VARIABLE = "radiation_frequency"
_RANGE_VARIABLE = "range"


def _make_dims() -> List[str]:
    return [_RANGE_VARIABLE, _FREQUENCY_VARIABLE]


def _make_frequency(header: CSFileHeader) -> xr.Variable:
    return xr.Variable(
        dims=_FREQUENCY_VARIABLE,
        data=1e6 * header.rep_freq_mhz
        + 1e3
        * header.bandwidth_khz
        * np.linspace(-0.5, 0.5, num=header.num_doppler_cells, endpoint=True),
        attrs=dict(units="Hz"),
    )


def _make_distance(header: CSFileHeader) -> xr.Variable:
    return xr.Variable(
        dims=_RANGE_VARIABLE,
        data=1000.0
        * header.range_cell_dist_km
        * np.arange(
            0,
            header.num_range_cells,
        ),
        attrs=dict(units="m"),
    )


def to_xarray(header: CSFileHeader, spectrum: Spectrum) -> xr.Dataset:
    assert header.version >= 4

    return xr.Dataset(
        data_vars=dict(
            antenna1=xr.DataArray(
                data=spectrum.antenna1,
                dims=_make_dims(),
                attrs=dict(
                    description="Antenna 1 Range-Dependent Power Spectral Density",
                ),
            ),
            antenna2=xr.DataArray(
                data=spectrum.antenna2,
                dims=_make_dims(),
                attrs=dict(
                    description="Antenna 2 Range-Dependent Power Spectral Density",
                ),
            ),
            antenna3=xr.DataArray(
                data=spectrum.antenna3,
                dims=_make_dims(),
                attrs=dict(
                    description="Antenna 3 Range-Dependent Power Spectral Density",
                ),
            ),
            cross12=xr.DataArray(
                data=spectrum.cross12,
                dims=_make_dims(),
                attrs=dict(
                    description="Relative phase between antenna 1 and 2"
                ),
            ),
            cross23=xr.DataArray(
                data=spectrum.cross23,
                dims=_make_dims(),
                attrs=dict(
                    description="Relative phase between antenna 2 and 3"
                ),
            ),
            cross13=xr.DataArray(
                data=spectrum.cross13,
                dims=_make_dims(),
                attrs=dict(
                    description="Relative phase between antenna 1 and 3"
                ),
            ),
        ),
        coords=dict(
            range=_make_distance(header),
            radiation_frequency=_make_frequency(header),
        ),
        attrs=dict(reference_time=header.timestamp),
    )
