from typing import Iterable

import numpy as np

from radarqc import csfile
from radarqc.csfile import CSFile, CSFileHeader
from radarqc.processing import SignalProcessor


class DataSet:
    """Supports aggregation of all Cross-Spectrum files in a given directory
    into a batch of images.

    Uses the monopole antenna channel (Antenna 3) for the spectrum"""

    def __init__(
        self, paths: Iterable[str], preprocess: SignalProcessor
    ) -> None:
        self._headers, spectra = [], []
        for cs in self._load_cs_files(paths, preprocess):
            self._headers.append(cs.header)
            spectra.append(cs.antenna3)
        self._spectra = np.stack(spectra)

    @property
    def spectra(self) -> np.ndarray:
        """Array size is (N, num_range, num_doppler), where N is the total
        number of Cross-Spectrum files found in the target directory"""
        return self._spectra

    @property
    def headers(self) -> Iterable[CSFileHeader]:
        """Returns an iterable containing the Cross-Spectrum file header
        for each input path"""
        return self._headers

    def _load_cs_files(
        self, paths: Iterable[str], preprocess: SignalProcessor
    ) -> Iterable[CSFile]:
        for path in paths:
            with open(path, "rb") as f:
                yield csfile.load(f, preprocess)
