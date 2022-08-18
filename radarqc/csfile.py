import io
from typing import BinaryIO, Optional

import numpy as np

from radarqc.header import CSFileHeader
from radarqc.xarray import to_xarray, xr
from radarqc.processing import Identity, SignalProcessor
from radarqc.reader import CSFileReader
from radarqc.writer import CSFileWriter
from radarqc.spectrum import Spectrum


class CSFile:
    """Representation of Cross-Spectrum file for storing CODAR HF radar data."""

    def __init__(self, header: CSFileHeader, spectrum: Spectrum) -> None:
        self._header = header
        self._spectrum = spectrum

    @property
    def header(self) -> CSFileHeader:
        """File header, contains all file metadata"""
        return self._header

    @property
    def spectrum(self) -> Spectrum:
        """Spectrum object containing all file metadata"""
        return self._spectrum

    @property
    def antenna1(self) -> np.ndarray:
        """Spectrum from first loop antenna"""
        return self._spectrum.antenna1

    @property
    def antenna2(self) -> np.ndarray:
        """Spectrum from second loop antenna"""
        return self._spectrum.antenna2

    @property
    def antenna3(self) -> np.ndarray:
        """Spectrum from monopole antenna"""
        return self._spectrum.antenna3

    @property
    def cross12(self) -> np.ndarray:
        """Cross-spectrum from antenna 1 & 2."""
        return self._spectrum.cross12

    @property
    def cross13(self) -> np.ndarray:
        """Cross-spectrum from antenna 1 & 3."""
        return self._spectrum.cross13

    @property
    def cross23(self) -> np.ndarray:
        """Cross-spectrum from antenna 2 & 3."""
        return self._spectrum.cross23

    def to_xarray(self) -> xr.Dataset:
        return to_xarray(self.header, self.spectrum)


def load(
    file: BinaryIO, preprocess: Optional[SignalProcessor] = None
) -> CSFile:
    """Deserialize file (a .read()-supporting binary file-like object
    containing cross-spectrum file data) to a Python object.

    preprocess is an optional SignalProcessor object that will be called with
    the result of the deserialization. The return value of preprocess will be
    used instead of the original object. This feature can be used to implement
    custom preprocessing or filtering of spectrum data."""
    if preprocess is None:
        preprocess = Identity()

    header, spectrum = CSFileReader().load(file, preprocess)
    return CSFile(header, spectrum)


def loads(
    archive: bytes, preprocess: Optional[SignalProcessor] = None
) -> CSFile:
    """Deserialize archive (bytes instance containing cross-spectrum file data)
    to a Python object.

    preprocess is an optional SignalProcessor object that will be called with
    the result of the deserialization. The return value of preprocess will be
    used instead of the original object. This feature can be used to implement
    custom preprocessing or filtering of spectrum data."""
    return load(io.BytesIO(archive), preprocess)


def dump(obj: CSFile, file: BinaryIO) -> None:
    """
    Serialize obj as a cross-spectra binary archive to file (a
    .write()-supporting binary file-like object)."""
    header, spectrum = obj.header, obj.spectrum
    CSFileWriter().dump(header, spectrum, file)


def dumps(obj: CSFile) -> bytes:
    """Serialize obj to a cross-spectra binary archive"""
    archive = io.BytesIO()
    dump(obj, archive)
    return archive.getbuffer()
