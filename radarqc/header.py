import datetime

from typing import Any
from collections import OrderedDict


class CSFileHeader:
    """Stores all header information from Cross-Spectrum files"""

    def __init__(self) -> None:
        self.__dict = OrderedDict()
        self.version: int = None
        self.timestamp: datetime.datetime = None
        self.cskind: int = None
        self.site_code: str = None
        self.cover_minutes: int = None
        self.deleted_source: bool = None
        self.override_source: bool = None
        self.start_freq_mhz: float = None
        self.rep_freq_mhz: float = None
        self.bandwidth_khz: float = None
        self.sweep_up: bool = None
        self.num_doppler_cells: int = None
        self.num_range_cells: int = None
        self.first_range_cell: int = None
        self.range_cell_dist_km: float = None
        self.output_interval: int = None
        self.create_type_code: str = None
        self.creator_version: str = None
        self.num_active_channels: int = None
        self.num_spectra_channels: int = None
        self.active_channels: int = None
        self.blocks = OrderedDict()

    def __setattr__(self, attr: str, value: Any) -> None:
        if attr == "_CSFileHeader__dict":
            super().__setattr__(attr, value)
        else:
            self.__dict[attr] = value

    def __getattr__(self, attr: str) -> Any:
        if attr == "_CSFileHeader__dict":
            return super().__getattr__(attr)
        else:
            return self.__dict[attr]

    def __repr__(self) -> str:
        title = "Class {}:".format(self.__class__.__name__)
        fields = (
            "- {:24s} {}".format(k, v)
            for k, v in self.__dict.items()
            if k != "blocks"
        )
        blocks = ("- {:24s} {}".format(k, v) for k, v in self.blocks.items())
        return "\n".join((title, *fields, *blocks))
