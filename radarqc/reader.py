import abc
import datetime

from typing import BinaryIO, Tuple

import numpy as np

from radarqc.header import CSFileHeader
from radarqc.processing import SignalProcessor
from radarqc.registry import ClassRegistry
from radarqc.serialization import BinaryReader, ByteOrder
from radarqc.spectrum import Spectrum


class _CSBlockReader(abc.ABC):
    __registry = ClassRegistry()

    def __init_subclass__(cls, /, tag: str, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        _CSBlockReader.__registry.register(tag, _CSBlockReader, cls)

    @staticmethod
    def make(tag: str):
        return _CSBlockReader.__registry.make(tag, _RawBlockReader)

    def read_block(
        self, reader: BinaryReader, block_size: int, header: CSFileHeader
    ):
        return self._read_block(reader, block_size, header)

    @abc.abstractmethod
    def _read_block(
        self, reader: BinaryReader, block_size: int, header: CSFileHeader
    ):
        """Subclasses will represent different blocks"""


class _RawBlockReader(_CSBlockReader, tag=None):
    def _read_block(
        self, reader: BinaryReader, block_size: int, header: CSFileHeader
    ):
        return reader.read_bytes(block_size)


class _CSBlockReaderTIME(_CSBlockReader, tag="TIME"):
    def _read_block(
        self, reader: BinaryReader, block_size: int, header: CSFileHeader
    ) -> dict:
        time_mark = reader.read_uint8()
        year = reader.read_uint16()
        month, day, hour, minute = reader.read_uint8(4)
        seconds, coverage_seconds, hours_from_utc = reader.read_double(3)
        return {
            "time_mark": time_mark,
            "year": year,
            "month": month,
            "day": day,
            "hour": hour,
            "minute": minute,
            "seconds": seconds,
            "coverage_seconds": coverage_seconds,
            "hours_from_utc": hours_from_utc,
        }


class _CSBlockReaderZONE(_CSBlockReader, tag="ZONE"):
    def _read_block(
        self, reader: BinaryReader, block_size: int, header: CSFileHeader
    ) -> str:
        return reader.read_string(block_size)


class _CSBlockReaderCITY(_CSBlockReader, tag="CITY"):
    def _read_block(
        self, reader: BinaryReader, block_size: int, header: CSFileHeader
    ) -> str:
        return reader.read_string(block_size)


class _CSBlockReaderLOCA(_CSBlockReader, tag="LOCA"):
    def _read_block(
        self, reader: BinaryReader, block_size: int, header: CSFileHeader
    ):
        latitude, longitude, altitude = reader.read_double(3)
        return {
            "latitude": latitude,
            "longitude": longitude,
            "altitude_meters": altitude,
        }


class _CSBlockReaderSITD(_CSBlockReader, tag="SITD"):
    def _read_block(
        self, reader: BinaryReader, block_size: int, header: CSFileHeader
    ) -> str:
        return reader.read_string(block_size)


class _CSBlockReaderRCVI(_CSBlockReader, tag="RCVI"):
    def _read_block(
        self, reader: BinaryReader, block_size: int, header: CSFileHeader
    ) -> dict:
        receiver_model, antenna_model = reader.read_uint32(2)
        reference_gain_db = reader.read_double()
        firmware = reader.read_string(32)
        return {
            "receiver_model": receiver_model,
            "antenna_model": antenna_model,
            "reference_gain_db": reference_gain_db,
            "firmware": firmware,
        }


class _CSBlockReaderTOOL(_CSBlockReader, tag="TOOL"):
    def _read_block(
        self, reader: BinaryReader, block_size: int, header: CSFileHeader
    ) -> str:
        return reader.read_string(block_size)


class _CSBlockReaderGLRM(_CSBlockReader, tag="GLRM"):
    def _read_block(
        self, reader: BinaryReader, block_size: int, header: CSFileHeader
    ) -> dict:
        method, version = reader.read_uint8(2)
        (
            num_points_removed,
            num_times_removed,
            num_segments_removed,
        ) = reader.read_uint32(3)
        (
            point_power_threshold,
            range_power_threshold,
            range_bin_threshold,
        ) = reader.read_double(3)
        remove_dc = bool(reader.read_uint8())
        return {
            "method": method,
            "version": version,
            "num_points_removed": num_points_removed,
            "num_times_removed": num_times_removed,
            "num_segments_removed": num_segments_removed,
            "point_power_threshold": point_power_threshold,
            "range_power_threshold": range_power_threshold,
            "range_bin_threshold": range_bin_threshold,
            "remove_dc": remove_dc,
        }


class _CSBlockReaderSUPI(_CSBlockReader, tag="SUPI"):
    def _read_block(
        self, reader: BinaryReader, block_size: int, header: CSFileHeader
    ):
        return reader.read_bytes(block_size)


class _CSBlockReaderSUPM(_CSBlockReader, tag="SUPM"):
    def _read_block(
        self, reader: BinaryReader, block_size: int, header: CSFileHeader
    ):
        return reader.read_bytes(block_size)


class _CSBlockReaderSUPP(_CSBlockReader, tag="SUPP"):
    def _read_block(
        self, reader: BinaryReader, block_size: int, header: CSFileHeader
    ):
        return reader.read_bytes(block_size)


class _CSBlockReaderANTG(_CSBlockReader, tag="ANTG"):
    def _read_block(
        self, reader: BinaryReader, block_size: int, header: CSFileHeader
    ):
        return reader.read_bytes(block_size)


class _CSBlockReaderFWIN(_CSBlockReader, tag="FWIN"):
    def _read_block(
        self, reader: BinaryReader, block_size: int, header: CSFileHeader
    ):
        return reader.read_bytes(block_size)


class _CSBlockReaderIQAP(_CSBlockReader, tag="IQAP"):
    def _read_block(
        self, reader: BinaryReader, block_size: int, header: CSFileHeader
    ):
        return reader.read_bytes(block_size)


class _CSBlockReaderFILL(_CSBlockReader, tag="FILL"):
    def _read_block(
        self, reader: BinaryReader, block_size: int, header: CSFileHeader
    ):
        return reader.read_bytes(block_size)


class _CSBlockReaderFOLS(_CSBlockReader, tag="FOLS"):
    def _read_block(
        self, reader: BinaryReader, block_size: int, header: CSFileHeader
    ):
        return [reader.read_int32(4) for _ in range(header.num_range_cells)]


class _CSBlockReaderWOLS(_CSBlockReader, tag="WOLS"):
    def _read_block(
        self, reader: BinaryReader, block_size: int, header: CSFileHeader
    ):
        return reader.read_bytes(block_size)


class _CSBlockReaderBRGR(_CSBlockReader, tag="BRGR"):
    def _read_block(
        self, reader: BinaryReader, block_size: int, header: CSFileHeader
    ):
        return reader.read_bytes(block_size)


class _CSBlockReaderEND6(_CSBlockReader, tag="END6"):
    def _read_block(
        self, reader: BinaryReader, block_size: int, header: CSFileHeader
    ) -> str:
        return reader.read_string(block_size)


class CSFileReader:
    """Responsible for parsing binary data encoded in Cross-Spectrum files"""

    def load(
        self, f: BinaryIO, preprocess: SignalProcessor
    ) -> Tuple[CSFileHeader, Spectrum]:
        reader = BinaryReader(f, ByteOrder.NETWORK)
        header = self._read_header(reader, version=reader.read_int16())
        spectrum = self._read_spectrum(reader, header, preprocess)
        return header, spectrum

    def _parse_timestamp(self, seconds: int) -> datetime.datetime:
        start = datetime.datetime(year=1904, month=1, day=1)
        return start + datetime.timedelta(seconds=seconds)

    def _get_block_parser(self, block_key: str) -> _CSBlockReader:
        return _CSBlockReader.make(block_key)

    def _read_header_bytes_v1(
        self, reader: BinaryReader, header: CSFileHeader
    ) -> None:
        header.timestamp = self._parse_timestamp(reader.read_uint32())
        reader.read_int32()  # v1_extent

    def _read_header_bytes_v2(
        self, reader: BinaryReader, header: CSFileHeader
    ) -> None:
        header.cskind = reader.read_int16()
        reader.read_int32()  # v2_extent

    def _read_header_bytes_v3(
        self, reader: BinaryReader, header: CSFileHeader
    ) -> None:
        header.site_code = reader.read_string(4)
        reader.read_int32()  # v3_extent

    def _read_header_bytes_v4(
        self, reader: BinaryReader, header: CSFileHeader
    ) -> None:
        header.cover_minutes = reader.read_int32()
        header.deleted_source = bool(reader.read_int32())
        header.override_source = bool(reader.read_int32())
        header.start_freq_mhz = reader.read_float()
        header.rep_freq_mhz = reader.read_float()
        header.bandwidth_khz = reader.read_float()
        header.sweep_up = bool(reader.read_int32())
        header.num_doppler_cells = reader.read_int32()
        header.num_range_cells = reader.read_int32()
        header.first_range_cell = reader.read_int32()
        header.range_cell_dist_km = reader.read_float()
        reader.read_int32()  # v4_extent

    def _read_header_bytes_v5(
        self, reader: BinaryReader, header: CSFileHeader
    ) -> None:
        header.output_interval = reader.read_int32()
        header.create_type_code = reader.read_string(4)
        header.creator_version = reader.read_string(4)
        header.num_active_channels = reader.read_int32()
        header.num_spectra_channels = reader.read_int32()
        header.active_channels = reader.read_uint32()
        reader.read_int32()  # v5_extent

    def _read_header_bytes_v6(
        self, reader: BinaryReader, header: CSFileHeader
    ) -> None:
        section_size = reader.read_uint32()
        while section_size > 0:
            block_key = reader.read_string(4)
            block_size = reader.read_uint32()

            parser = self._get_block_parser(block_key)
            block = parser.read_block(reader, block_size, header)
            header.blocks[block_key] = block

            section_size -= 8  # Bytes corresponding to block key and size
            section_size -= block_size

    def _read_header(self, reader: BinaryReader, version: int) -> CSFileHeader:
        header = CSFileHeader(version=version)
        if version < 1 or version > 6:
            raise ValueError(
                "Detected file version lies outside of accepted range"
            )

        self._read_header_bytes_v1(reader, header)
        if version == 1:  # Detected file version v1
            return header

        self._read_header_bytes_v2(reader, header)
        if version == 2:  # Detected file version v2
            return header

        self._read_header_bytes_v3(reader, header)
        if version == 3:  # Detected file version v3
            return header

        self._read_header_bytes_v4(reader, header)
        if version == 4:  # Detected file version v4
            return header

        self._read_header_bytes_v5(reader, header)
        if version == 5:  # Detected file version v5
            return header

        self._read_header_bytes_v6(reader, header)
        return header  # File version must be v6

    def _read_spectrum(
        self,
        reader: BinaryReader,
        header: CSFileHeader,
        preprocess: SignalProcessor,
    ) -> Spectrum:
        a1, a2, a3, c12, c13, c23, q = [], [], [], [], [], [], []
        for _ in range(header.num_range_cells):
            a1.append(self._read_real_row(reader, header))
            a2.append(self._read_real_row(reader, header))
            a3.append(self._read_real_row(reader, header))
            c12.append(self._read_complex_row(reader, header))
            c13.append(self._read_complex_row(reader, header))
            c23.append(self._read_complex_row(reader, header))
            if header.cskind >= 2:
                q.append(self._read_real_row(reader, header))

        return Spectrum(
            np.stack(a1),
            np.stack(a2),
            np.stack(a3),
            np.stack(c12),
            np.stack(c13),
            np.stack(c23),
            np.stack(q),
            preprocess,
        )

    def _read_real_row(
        self, reader: BinaryReader, header: CSFileHeader
    ) -> np.ndarray:
        floats = reader.read_float(header.num_doppler_cells)
        return np.array(floats, dtype=np.float32)

    def _read_complex_row(
        self, reader: BinaryReader, header: CSFileHeader
    ) -> np.ndarray:
        floats = reader.read_float(header.num_doppler_cells * 2)
        return np.array(floats, dtype=np.float32).view(np.complex64)
