from dataclasses import dataclass, field
from typing import Optional
import datetime

from collections import OrderedDict


@dataclass
class CSFileHeader:
    """Stores all header information from Cross-Spectrum files"""

    version: int  # Version is the only required argument
    timestamp: Optional[datetime.datetime] = None
    cskind: Optional[int] = None
    site_code: Optional[str] = None
    cover_minutes: Optional[int] = None
    deleted_source: Optional[bool] = None
    override_source: Optional[bool] = None
    start_freq_mhz: Optional[float] = None
    rep_freq_mhz: Optional[float] = None
    bandwidth_khz: Optional[float] = None
    sweep_up: Optional[bool] = None
    num_doppler_cells: Optional[int] = None
    num_range_cells: Optional[int] = None
    first_range_cell: Optional[int] = None
    range_cell_dist_km: Optional[float] = None
    output_interval: Optional[int] = None
    create_type_code: Optional[str] = None
    creator_version: Optional[str] = None
    num_active_channels: Optional[int] = None
    num_spectra_channels: Optional[int] = None
    active_channels: Optional[int] = None
    blocks: OrderedDict = field(default_factory=OrderedDict)
