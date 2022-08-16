import struct
import enum

from typing import Any, BinaryIO, Iterable, Union


class ByteOrder(enum.Enum):
    BIG_ENDIAN = 1
    LITTLE_ENDIAN = 2
    NATIVE = 3
    NETWORK = 4


class _Formatter:
    _BYTE_ORDER_CHAR = {
        ByteOrder.BIG_ENDIAN: ">",
        ByteOrder.LITTLE_ENDIAN: "<",
        ByteOrder.NATIVE: "=",
        ByteOrder.NETWORK: "!",
    }

    def __init__(self, byteorder: ByteOrder) -> None:
        self._byteorder = self._BYTE_ORDER_CHAR[byteorder]

    def create_format(self, fmt: str, n: int) -> str:
        return "{}{}".format(self._byteorder, n * fmt)


class BinaryReader:
    def __init__(self, f: BinaryIO, byteorder: ByteOrder) -> None:
        self._file = f
        self._formatter = _Formatter(byteorder)

    def read_string(self, n: int = 1) -> str:
        raw = self.read_char(n=n)
        if n == 1:
            return raw.decode()
        return "".join(r.decode() for r in raw)

    def read_bytes(self, n: int = 1) -> bytes:
        raw = self.read_char(n=n)
        if n == 1:
            return raw
        return b"".join(raw)

    def read_bool(self, n: int = 1) -> Union[bool, Iterable[bool]]:
        return self._read("?", size=1, n=n)

    def read_char(self, n: int = 1) -> Union[bytes, Iterable[bytes]]:
        return self._read("c", size=1, n=n)

    def read_int8(self, n: int = 1) -> Union[int, Iterable[int]]:
        return self._read("b", size=1, n=n)

    def read_uint8(self, n: int = 1) -> Union[int, Iterable[int]]:
        return self._read("B", size=1, n=n)

    def read_int16(self, n: int = 1) -> Union[int, Iterable[int]]:
        return self._read("h", size=2, n=n)

    def read_uint16(self, n: int = 1) -> Union[int, Iterable[int]]:
        return self._read("H", size=2, n=n)

    def read_int32(self, n: int = 1) -> Union[int, Iterable[int]]:
        return self._read("i", size=4, n=n)

    def read_uint32(self, n: int = 1) -> Union[int, Iterable[int]]:
        return self._read("I", size=4, n=n)

    def read_int64(self, n: int = 1) -> Union[int, Iterable[int]]:
        return self._read("q", size=8, n=n)

    def read_uint64(self, n: int = 1) -> Union[int, Iterable[int]]:
        return self._read("Q", size=8, n=n)

    def read_float(self, n: int = 1) -> Union[float, Iterable[float]]:
        return self._read("f", size=4, n=n)

    def read_double(self, n: int = 1) -> Union[float, Iterable[float]]:
        return self._read("d", size=8, n=n)

    def _read(self, fmt: str, size: int, n: int) -> Any:
        return self._unpack_bytes(fmt, self._file.read(size * n), n)

    def _unpack_bytes(self, fmt: str, buff: bytes, n: int) -> Any:
        data = struct.unpack(self._formatter.create_format(fmt, n), buff)
        if n == 1:
            return data[0]
        return data


class BinaryWriter:
    def __init__(self, f: BinaryIO, byteorder: ByteOrder) -> None:
        self._file = f
        self._formatter = _Formatter(byteorder)

    def write_string(self, buff: str) -> None:
        self._write_bytes(buff.encode())

    def write_bytes(self, buff: bytes) -> None:
        self._write_bytes(buff)

    def write_char(self, buff: Union[bytes, Iterable[bytes]]) -> None:
        return self._write(buff, "c")

    def write_int8(self, buff: Union[int, Iterable[int]]) -> None:
        return self._write(buff, "b")

    def write_uint8(self, buff: Union[int, Iterable[int]]) -> None:
        return self._write(buff, "B")

    def write_bool(self, buff: Union[int, Iterable[int]]) -> None:
        return self._write(buff, "?")

    def write_int16(self, buff: Union[int, Iterable[int]]) -> None:
        return self._write(buff, "h")

    def write_uint16(self, buff: Union[int, Iterable[int]]) -> None:
        return self._write(buff, "H")

    def write_int32(self, buff: Union[int, Iterable[int]]) -> None:
        return self._write(buff, "i")

    def write_uint32(self, buff: Union[int, Iterable[int]]) -> None:
        return self._write(buff, "I")

    def write_int64(self, buff: Union[int, Iterable[int]]) -> None:
        return self._write(buff, "q")

    def write_uint64(self, buff: Union[int, Iterable[int]]) -> None:
        return self._write(buff, "Q")

    def write_float(self, buff: Union[float, Iterable[float]]) -> None:
        return self._write(buff, "f")

    def write_double(self, buff: Union[float, Iterable[float]]) -> None:
        return self._write(buff, "d")

    def _write(self, data: Any, fmt: str) -> Any:
        self._file.write(self._pack_bytes(data, fmt))

    def _pack_bytes(self, data: Any, fmt: str) -> bytes:
        if hasattr(data, "__len__"):
            return struct.pack(
                self._formatter.create_format(fmt, len(data)), *data
            )
        return struct.pack(self._formatter.create_format(fmt, 1), data)
