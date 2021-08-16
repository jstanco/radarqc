import abc
import io
import datetime
import struct

from collections import defaultdict
from typing import BinaryIO, Tuple

import numpy as np

from radarqc.header import CSFileHeader
from radarqc.processing import SignalProcessor
from radarqc.serialization import BinaryReader, ByteOrder
from radarqc.spectrum import Spectrum


class ClassRegistryError(Exception):
    """Custom exception handling case where multiple classes have been
    registered to handle deserializing the same block tag"""

    def __init__(self, subclass: type, tag: str) -> None:
        message = (
            "Cannot register two CSBlockReader subclasses for tag '{}'.".format(
                tag
            )
        )
        super().__init__(message)
        self.subclass = subclass
        self.tag = tag


class _CSBlockReader(abc.ABC):
    __readers = {}

    def __init_subclass__(cls, /, tag: str, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        try:
            _CSBlockReader.__readers[tag]
        except KeyError:
            _CSBlockReader.__readers[tag] = cls
        else:
            raise ClassRegistryError(
                subclass=cls,
                tag=tag,
            )

    @staticmethod
    def __getreadercls(tag: str) -> type:
        return _CSBlockReader.__readers.get(tag, _RawBlockReader)

    @staticmethod
    def create(tag: str):
        reader_cls = _CSBlockReader.__getreadercls(tag)
        return reader_cls()

    def read_block(
        self, reader: BinaryReader, block_size: int, header: CSFileHeader
    ):
        return self._read_block(reader, block_size, header)

    @abc.abstractmethod
    def _read_block(
        self, reader: BinaryReader, block_size: int, header: CSFileHeader
    ):
        """Subclasses will represent different blocks"""


class _CSBlockReaderTIME(_CSBlockReader, tag="TIME"):
    def _read_block(
        self, reader: BinaryReader, block_size: int, header: CSFileHeader
    ):
        return reader.read_bytes(block_size)


class _CSBlockReaderZONE(_CSBlockReader, tag="ZONE"):
    def _read_block(
        self, reader: BinaryReader, block_size: int, header: CSFileHeader
    ):
        return reader.read_bytes(block_size)


class _CSBlockReaderLOCA(_CSBlockReader, tag="LOCA"):
    def _read_block(
        self, reader: BinaryReader, block_size: int, header: CSFileHeader
    ):
        return reader.read_bytes(block_size)


class _CSBlockReaderRCVI(_CSBlockReader, tag="RCVI"):
    def _read_block(
        self, reader: BinaryReader, block_size: int, header: CSFileHeader
    ):
        return reader.read_bytes(block_size)


class _CSBlockReaderGLRM(_CSBlockReader, tag="GLRM"):
    def _read_block(
        self, reader: BinaryReader, block_size: int, header: CSFileHeader
    ):
        return reader.read_bytes(block_size)


class _CSBlockReaderFOLS(_CSBlockReader, tag="FOLS"):
    def _read_block(
        self, reader: BinaryReader, block_size: int, header: CSFileHeader
    ):
        return reader.read_bytes(block_size)


class _CSBlockReaderEND6(_CSBlockReader, tag="END6"):
    def _read_block(
        self, reader: BinaryReader, block_size: int, header: CSFileHeader
    ):
        reader.read_bytes(block_size)
        return b"".decode()


class _RawBlockReader(_CSBlockReader, tag=None):
    def _read_block(
        self, reader: BinaryReader, block_size: int, header: CSFileHeader
    ):
        return reader.read_bytes(block_size)


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
