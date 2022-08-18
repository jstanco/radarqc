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


_FREQUENCY_NAME = "radiation_frequency"
_RANGE_NAME = "range"
_FREQUENCY_UNITS = "Hz"
_RANGE_UNITS = "m"
_ONE_MHZ = 1e6
_ONE_KHZ = 1e3


def _make_dims() -> List[str]:
    return [_RANGE_NAME, _FREQUENCY_NAME]


def _make_frequency(header: CSFileHeader) -> xr.Variable:
    start_frequency = _ONE_MHZ * header.start_freq_mhz
    delta_frequency = (
        _ONE_KHZ
        * header.bandwidth_khz
        * (
            np.linspace(0.0, 1.0, num=header.num_doppler_cells, endpoint=True)
            if header.sweep_up
            else np.linspace(
                -1.0, 0.0, num=header.num_doppler_cells, endpoint=True
            )
        )
    )

    return xr.Variable(
        dims=_FREQUENCY_NAME,
        data=start_frequency + delta_frequency,
        attrs=dict(units=_FREQUENCY_UNITS),
    )


def _make_distance(header: CSFileHeader) -> xr.Variable:
    return xr.Variable(
        dims=_RANGE_NAME,
        data=1000.0
        * header.range_cell_dist_km
        * (
            np.arange(
                0,
                header.num_range_cells,
            )
            + (header.first_range_cell - 1)
        ),
        attrs=dict(units=_RANGE_UNITS),
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
        attrs=dict(
            timestamp=header.timestamp,
            site_code=header.site_code,
            cover_minutes=header.cover_minutes,
            deleted_source=header.deleted_source,
            override_source=header.override_source,
            rep_freq_hz=header.rep_freq_hz,
            output_interval=header.output_interval,
            create_type_code=header.create_type_code,
            creator_version=header.creator_version,
            num_active_channels=header.num_active_channels,
            num_spectra_channels=header.num_spectra_channels,
            active_channels=header.active_channels,
            **header.blocks
        ),
    )
