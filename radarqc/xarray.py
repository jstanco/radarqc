from typing import List
import warnings

import numpy as np
import json

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
_COMPLEX_NAME = "complex"
_FREQUENCY_UNITS = "Hz"
_RANGE_UNITS = "m"
_ONE_MHZ = 1e6
_ONE_KHZ = 1e3


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

    frequencies = start_frequency + delta_frequency
    return xr.Variable(
        dims=_FREQUENCY_NAME,
        data=frequencies.astype(np.float32),
        attrs=dict(units=_FREQUENCY_UNITS),
    )


def _make_range(header: CSFileHeader) -> xr.Variable:
    ranges = (
        1000.0
        * header.range_cell_dist_km
        * (
            np.arange(
                0,
                header.num_range_cells,
            )
            + (header.first_range_cell - 1)
        )
    )

    return xr.Variable(
        dims=_RANGE_NAME,
        data=ranges.astype(np.float32),
        attrs=dict(units=_RANGE_UNITS),
    )


def _complex_as_real(x: np.ndarray) -> np.ndarray:
    return x.view(dtype=np.float32).reshape(*x.shape, 2)


def _make_real_array(x: np.ndarray, description: str) -> xr.DataArray:
    return xr.DataArray(
        data=x,
        dims=[_RANGE_NAME, _FREQUENCY_NAME],
        attrs=dict(description=description),
    )


def _make_complex_array(x: np.ndarray, description: str) -> xr.DataArray:
    return xr.DataArray(
        data=_complex_as_real(x),
        dims=[_RANGE_NAME, _FREQUENCY_NAME, _COMPLEX_NAME],
        attrs=dict(description=description),
    )


def to_xarray(header: CSFileHeader, spectrum: Spectrum) -> xr.Dataset:
    assert header.version >= 4

    return xr.Dataset(
        data_vars=dict(
            antenna1=_make_real_array(
                spectrum.antenna1,
                description="Antenna 1 Range-Dependent Self-Spectrum",
            ),
            antenna2=_make_real_array(
                spectrum.antenna2,
                description="Antenna 2 Range-Dependent Self-Spectrum",
            ),
            antenna3=_make_real_array(
                spectrum.antenna3,
                description="Antenna 3 Range-Dependent Self-Spectrum",
            ),
            cross12=_make_complex_array(
                spectrum.cross12,
                description="Cross-Spectrum between antennae 1 and 2",
            ),
            cross23=_make_complex_array(
                spectrum.cross23,
                description="Cross-Spectrum between antennae 2 and 3",
            ),
            cross13=_make_complex_array(
                spectrum.cross13,
                description="Cross-Spectrum between antennae 1 and 3",
            ),
        ),
        coords=dict(
            range=_make_range(header),
            radiation_frequency=_make_frequency(header),
        ),
        attrs=dict(
            timestamp=header.timestamp.isoformat(),
            site_code=header.site_code,
            cover_minutes=header.cover_minutes,
            deleted_source=int(header.deleted_source),
            override_source=int(header.override_source),
            rep_freq_hz=header.rep_freq_hz,
            output_interval=header.output_interval,
            create_type_code=header.create_type_code,
            creator_version=header.creator_version,
            num_active_channels=header.num_active_channels,
            num_spectra_channels=header.num_spectra_channels,
            active_channels=header.active_channels,
            blocks=json.dumps(
                header.blocks
            ),  # TODO (John): Define custom block handlers
        ),
    )
