import abc
import io
import datetime

from typing import Any, BinaryIO

import numpy as np

from radarqc.header import CSFileHeader
from radarqc.registry import ClassRegistry
from radarqc.serialization import BinaryWriter, ByteOrder
from radarqc.spectrum import Spectrum


class _CSBlockWriter(abc.ABC):
    __registry = ClassRegistry()

    def __init_subclass__(cls, /, tag: str, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        _CSBlockWriter.__registry.register(tag, _CSBlockWriter, cls)

    @staticmethod
    def make(tag: str):
        return _CSBlockWriter.__registry.make(tag, _RawBlockWriter)

    def write_block(self, writer: BinaryWriter, block: Any) -> None:
        return self._write_block(writer, block)

    @abc.abstractmethod
    def _write_block(self, writer: BinaryWriter, block: Any) -> None:
        """Subclasses will represent different blocks"""


class _RawBlockWriter(_CSBlockWriter, tag=None):
    def _write_block(self, writer: BinaryWriter, block: bytes) -> None:
        return writer.write_bytes(block)


class _CSBlockWriterTIME(_CSBlockWriter, tag="TIME"):
    def _write_block(self, writer: BinaryWriter, block: bytes) -> None:
        writer.write_uint8(block["time_mark"])
        writer.write_uint16(block["year"])
        writer.write_uint8(block["month"])
        writer.write_uint8(block["day"])
        writer.write_uint8(block["hour"])
        writer.write_uint8(block["minute"])
        writer.write_double(block["seconds"])
        writer.write_double(block["coverage_seconds"])
        writer.write_double(block["hours_from_utc"])


class _CSBlockWriterZONE(_CSBlockWriter, tag="ZONE"):
    def _write_block(self, writer: BinaryWriter, block: str) -> None:
        return writer.write_string(block)


class _CSBlockWriterCITY(_CSBlockWriter, tag="CITY"):
    def _write_block(self, writer: BinaryWriter, block: str) -> None:
        return writer.write_string(block)


class _CSBlockWriterLOCA(_CSBlockWriter, tag="LOCA"):
    def _write_block(self, writer: BinaryWriter, block: dict) -> None:
        writer.write_double(block["latitude"])
        writer.write_double(block["longitude"])
        writer.write_double(block["altitude_meters"])


class _CSBlockWriterSITD(_CSBlockWriter, tag="SITD"):
    def _write_block(self, writer: BinaryWriter, block: str) -> None:
        return writer.write_string(block)


class _CSBlockWriterRCVI(_CSBlockWriter, tag="RCVI"):
    def _write_block(self, writer: BinaryWriter, block: dict) -> None:
        writer.write_uint32(block["receiver_model"])
        writer.write_uint32(block["antenna_model"])
        writer.write_double(block["reference_gain_db"])
        writer.write_string(block["firmware"])


class _CSBlockWriterTOOL(_CSBlockWriter, tag="TOOL"):
    def _write_block(self, writer: BinaryWriter, block: str) -> None:
        return writer.write_string(block)


class _CSBlockWriterGLRM(_CSBlockWriter, tag="GLRM"):
    def _write_block(self, writer: BinaryWriter, block: dict) -> None:
        writer.write_uint8(block["method"])
        writer.write_uint8(block["version"])
        writer.write_uint32(block["num_points_removed"])
        writer.write_uint32(block["num_times_removed"])
        writer.write_uint32(block["num_segments_removed"])
        writer.write_double(block["point_power_threshold"])
        writer.write_double(block["range_power_threshold"])
        writer.write_double(block["range_bin_threshold"])
        writer.write_uint8(int(block["remove_dc"]))


class _CSBlockWriterSUPI(_CSBlockWriter, tag="SUPI"):
    def _write_block(self, writer: BinaryWriter, block: Any) -> None:
        return writer.write_bytes(block)


class _CSBlockWriterSUPM(_CSBlockWriter, tag="SUPM"):
    def _write_block(self, writer: BinaryWriter, block: Any) -> None:
        return writer.write_bytes(block)


class _CSBlockWriterSUPP(_CSBlockWriter, tag="SUPP"):
    def _write_block(self, writer: BinaryWriter, block: Any) -> None:
        return writer.write_bytes(block)


class _CSBlockWriterANTG(_CSBlockWriter, tag="ANTG"):
    def _write_block(self, writer: BinaryWriter, block: Any) -> None:
        return writer.write_bytes(block)


class _CSBlockWriterFWIN(_CSBlockWriter, tag="FWIN"):
    def _write_block(self, writer: BinaryWriter, block: Any) -> None:
        return writer.write_bytes(block)


class _CSBlockWriterIQAP(_CSBlockWriter, tag="IQAP"):
    def _write_block(self, writer: BinaryWriter, block: Any) -> None:
        return writer.write_bytes(block)


class _CSBlockWriterFILL(_CSBlockWriter, tag="FILL"):
    def _write_block(self, writer: BinaryWriter, block: Any) -> None:
        return writer.write_bytes(block)


class _CSBlockWriterFOLS(_CSBlockWriter, tag="FOLS"):
    def _write_block(self, writer: BinaryWriter, block: list) -> None:
        for indices in block:
            writer.write_int32(indices)


class _CSBlockWriterWOLS(_CSBlockWriter, tag="WOLS"):
    def _write_block(self, writer: BinaryWriter, block: Any) -> None:
        return writer.write_bytes(block)


class _CSBlockWriterBRGR(_CSBlockWriter, tag="BRGR"):
    def _write_block(self, writer: BinaryWriter, block: Any) -> None:
        return writer.write_bytes(block)


class _CSBlockWriterEND6(_CSBlockWriter, tag="END6"):
    def _write_block(self, writer: BinaryWriter, block: str) -> None:
        return writer.write_string(block)


class CSFileWriter:
    """Responsible for parsing binary data encoded in Cross-Spectrum files"""

    _V1_HEADER_SIZE = 10
    _V2_HEADER_SIZE = 16
    _V3_HEADER_SIZE = 24
    _V4_HEADER_SIZE = 72
    _V5_HEADER_SIZE = 100

    def dump(
        self, header: CSFileHeader, spectrum: Spectrum, f: BinaryIO
    ) -> None:
        writer = BinaryWriter(f, ByteOrder.BIG_ENDIAN)
        self._write_header(header, writer)
        self._write_spectrum_data(header, spectrum, writer)

    def _get_block_parser(self, block_key: str) -> _CSBlockWriter:
        return _CSBlockWriter.make(block_key)

    def _get_raw_timestamp(self, timestamp: datetime.datetime) -> int:
        start = datetime.datetime(year=1904, month=1, day=1)
        delta = timestamp - start
        return int(delta.total_seconds())

    def _calculate_block_section_size(self, blocks: dict) -> int:
        section_size = 0
        for block in blocks.values():
            section_size += 8  # Bytes corresponding to block key and size
            section_size += len(block)
        return section_size

    def _calculate_v1_extent(self, header_size: int) -> int:
        return header_size - self._V1_HEADER_SIZE

    def _calculate_v2_extent(self, header_size: int) -> int:
        return header_size - self._V2_HEADER_SIZE

    def _calculate_v3_extent(self, header_size: int) -> int:
        return header_size - self._V3_HEADER_SIZE

    def _calculate_v4_extent(self, header_size: int) -> int:
        return header_size - self._V4_HEADER_SIZE

    def _calculate_v5_extent(self, header_size: int) -> int:
        return header_size - self._V5_HEADER_SIZE

    def _calculate_header_size(self, blocks: dict, version: int) -> int:
        if version == 1:
            return self._V1_HEADER_SIZE
        if version == 2:
            return self._V2_HEADER_SIZE
        if version == 3:
            return self._V3_HEADER_SIZE
        if version == 4:
            return self._V4_HEADER_SIZE
        if version == 5:
            return self._V5_HEADER_SIZE
        if version == 6:
            return (
                self._V5_HEADER_SIZE  # Size of previous V5 header
                + 4  # 32-bit integer indicating number of blocks
                + self._calculate_block_section_size(blocks)
            )

    def _serialize_blocks(self, header: CSFileHeader) -> dict:
        raw_blocks = {}
        for block_key, block in header.blocks.items():
            block_writer = self._get_block_parser(block_key)
            blockio = io.BytesIO()
            writer = BinaryWriter(blockio, ByteOrder.BIG_ENDIAN)
            block_writer.write_block(writer, block)
            raw_blocks[block_key] = blockio.getbuffer()
        return raw_blocks

    def _write_header_bytes_v1(
        self, writer: BinaryWriter, header: CSFileHeader, header_size: int
    ) -> None:
        writer.write_int16(header.version)
        writer.write_uint32(self._get_raw_timestamp(header.timestamp))
        writer.write_int32(self._calculate_v1_extent(header_size))

    def _write_header_bytes_v2(
        self, writer: BinaryWriter, header: CSFileHeader, header_size: int
    ) -> None:
        writer.write_int16(header.cskind)
        writer.write_int32(self._calculate_v2_extent(header_size))

    def _write_header_bytes_v3(
        self, writer: BinaryWriter, header: CSFileHeader, header_size: int
    ) -> None:
        writer.write_string(header.site_code)
        writer.write_int32(self._calculate_v3_extent(header_size))

    def _write_header_bytes_v4(
        self, writer: BinaryWriter, header: CSFileHeader, header_size: int
    ) -> None:
        writer.write_int32(header.cover_minutes)
        writer.write_int32(header.deleted_source)
        writer.write_int32(header.override_source)
        writer.write_float(header.start_freq_mhz)
        writer.write_float(header.rep_freq_mhz)
        writer.write_float(header.bandwidth_khz)
        writer.write_int32(header.sweep_up)
        writer.write_int32(header.num_doppler_cells)
        writer.write_int32(header.num_range_cells)
        writer.write_int32(header.first_range_cell)
        writer.write_float(header.range_cell_dist_km)
        writer.write_int32(self._calculate_v4_extent(header_size))

    def _write_header_bytes_v5(
        self, writer: BinaryWriter, header: CSFileHeader, header_size: int
    ) -> None:
        writer.write_int32(header.output_interval)
        writer.write_string(header.create_type_code)
        writer.write_string(header.creator_version)
        writer.write_int32(header.num_active_channels)
        writer.write_int32(header.num_spectra_channels)
        writer.write_uint32(header.active_channels)
        writer.write_int32(self._calculate_v5_extent(header_size))

    def _write_header_bytes_v6(
        self,
        writer: BinaryWriter,
        blocks: dict,
    ) -> None:
        writer.write_uint32(self._calculate_block_section_size(blocks))
        for block_key, block in blocks.items():
            writer.write_string(block_key)
            writer.write_uint32(len(block))
            writer.write_bytes(block)

    def _write_header(self, writer: BinaryWriter, header: CSFileHeader) -> None:
        if header.version < 1 or header.version > 6:
            raise ValueError(
                "Detected file version lies outside of accepted range"
            )

        blocks = self._serialize_blocks(header)
        header_size = self._calculate_header_size(blocks, header.version)
        self._write_header_bytes_v1(writer, header, header_size)
        if header.version == 1:  # Detected file version v1
            return

        self._write_header_bytes_v2(writer, header, header_size)
        if header.version == 2:  # Detected file version v2
            return

        self._write_header_bytes_v3(writer, header, header_size)
        if header.version == 3:  # Detected file version v3
            return

        self._write_header_bytes_v4(writer, header, header_size)
        if header.version == 4:  # Detected file version v4
            return

        self._write_header_bytes_v5(writer, header, header_size)
        if header.version == 5:  # Detected file version v5
            return

        # File version must be v6
        self._write_header_bytes_v6(writer, header, blocks)

    def _write_real_row(self, row: np.ndarray, writer: BinaryWriter) -> None:
        dtype = np.dtype("float32").newbyteorder(">")
        writer.write_bytes(row.astype(dtype=dtype))

    def _write_complex_row(self, row: np.ndarray, writer: BinaryWriter) -> None:
        writer.write_float(row.view(np.float32).tolist())

    def _write_spectrum_data(
        self, header: CSFileHeader, spectrum: Spectrum, writer: BinaryWriter
    ) -> None:
        for i in range(header.num_range_cells):
            self._write_real_row(spectrum.antenna1[i], writer)
            self._write_real_row(spectrum.antenna2[i], writer)
            self._write_real_row(spectrum.antenna3[i], writer)
            self._write_complex_row(spectrum.cross12[i], writer)
            self._write_complex_row(spectrum.cross13[i], writer)
            self._write_complex_row(spectrum.cross23[i], writer)
            if header.cskind >= 2:
                self._write_real_row(spectrum.quality[i], writer)
