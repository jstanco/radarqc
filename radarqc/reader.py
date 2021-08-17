import abc
import datetime
import struct

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
    def create(tag: str):
        return _CSBlockReader.__registry.create(tag, _RawBlockReader)

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
        seconds, coverage_seconds, from_utc = reader.read_double(3)
        return {
            "time_mark": time_mark,
            "year": year,
            "month": month,
            "day": day,
            "hour": hour,
            "minute": minute,
            "seconds": seconds,
            "coverage_seconds": coverage_seconds,
            "hours_from_utc": from_utc,
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
        lat, lon, alt = reader.read_double(3)
        return {"latitude": lat, "longitude": lon, "altitude_meters": alt}


class _CSBlockReaderSITD(_CSBlockReader, tag="SITD"):
    def _read_block(
        self, reader: BinaryReader, block_size: int, header: CSFileHeader
    ) -> str:
        return reader.read_string(block_size)


class _CSBlockReaderRCVI(_CSBlockReader, tag="RCVI"):
    def _read_block(
        self, reader: BinaryReader, block_size: int, header: CSFileHeader
    ) -> dict:
        recv_model, ant_model = reader.read_uint32(2)
        refgain_db = reader.read_double()
        firmware = reader.read_string(32)
        return {
            "receiver_model": recv_model,
            "antenna_model": ant_model,
            "reference_gain_db": refgain_db,
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
        points, times, segments = reader.read_uint32(3)
        point_thresh, range_thresh, bin_thresh = reader.read_double(3)
        remove_dc = bool(reader.read_uint8())
        return {
            "method": method,
            "version": version,
            "num_points_removed": points,
            "num_times_removed": times,
            "num_segments_removed": segments,
            "point_power_threshold": point_thresh,
            "range_power_threshold": range_thresh,
            "range_bin_threshold": bin_thresh,
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
        return self._read_cs_buff(f, preprocess)

    def _parse_timestamp(self, seconds: int) -> datetime.datetime:
        start = datetime.datetime(year=1904, month=1, day=1)
        delta = datetime.timedelta(seconds=seconds)
        return start + delta

    def _read_cs_buff(
        self, f: BinaryIO, preprocess: SignalProcessor
    ) -> Tuple[CSFileHeader, Spectrum]:
        readers = {6: self._read_buff_v6}
        version = self._read_version(f)
        unpack = readers[version]
        return unpack(f, preprocess)

    def _read_version(self, f: BinaryIO) -> int:
        version_size_bytes = 2
        buff = f.read(version_size_bytes)
        (version,) = struct.unpack_from(">h", buff)
        return version

    def _get_block_parser(self, block_key: str) -> _CSBlockReader:
        # return self._BLOCK_READERS[block_key]
        return _CSBlockReader.create(block_key)

    def _read_buff_v6(
        self, f: BinaryIO, preprocess: SignalProcessor
    ) -> Tuple[CSFileHeader, Spectrum]:
        reader = BinaryReader(f, ByteOrder.BIG_ENDIAN)
        header = self._read_header_v6(reader)
        spectrum = self._read_spectrum(reader, header, preprocess)
        return header, spectrum

    def _read_header_v6(self, reader: BinaryReader) -> CSFileHeader:
        header = CSFileHeader()
        header.version = 6
        header.timestamp = self._parse_timestamp(reader.read_uint32())
        reader.read_int32()  # v1_extent
        # end v1

        header.cskind = reader.read_int16()
        reader.read_int32()  # v2_extent
        # end v2

        header.site_code = reader.read_string(4)
        reader.read_int32()  # v3_extent
        # end v3

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
        # end v4

        header.output_interval = reader.read_int32()
        header.create_type_code = reader.read_string(4)
        header.creator_version = reader.read_string(4)
        header.num_active_channels = reader.read_int32()
        header.num_spectra_channels = reader.read_int32()
        header.active_channels = reader.read_uint32()
        reader.read_int32()  # v5_extent
        # end v5

        cs6_header_size = reader.read_uint32()
        while cs6_header_size > 0:
            block_key = reader.read_string(4)
            block_size = reader.read_uint32()

            parser = self._get_block_parser(block_key)
            block = parser.read_block(reader, block_size, header)
            header.blocks[block_key] = block

            cs6_header_size -= 8
            cs6_header_size -= block_size
        # end v6
        return header

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

        a1 = np.stack(a1)
        a2 = np.stack(a2)
        a3 = np.stack(a3)
        c12 = np.stack(c12)
        c13 = np.stack(c13)
        c23 = np.stack(c23)
        return Spectrum(a1, a2, a3, c12, c13, c23, q, preprocess)

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
