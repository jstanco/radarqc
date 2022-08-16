import numpy as np

from radarqc.processing import SignalProcessor


class Spectrum:
    """Stores antenna spectra from Cross-Spectrum files."""

    def __init__(
        self,
        antenna1: np.ndarray,
        antenna2: np.ndarray,
        antenna3: np.ndarray,
        cross12: np.ndarray,
        cross13: np.ndarray,
        cross23: np.ndarray,
        quality: np.ndarray,
        preprocess: SignalProcessor,
    ) -> None:
        self.antenna1 = preprocess(antenna1, preprocess)
        self.antenna2 = preprocess(antenna2, preprocess)
        self.antenna3 = preprocess(antenna3, preprocess)
        self.cross12 = self._preprocess_complex_signal(cross12, preprocess)
        self.cross13 = self._preprocess_complex_signal(cross13, preprocess)
        self.cross23 = self._preprocess_complex_signal(cross23, preprocess)
        self.quality = preprocess(quality, preprocess)

    def _preprocess_complex_signal(
        self, raw: np.ndarray, preprocess: SignalProcessor
    ) -> None:
        real, imag = preprocess(raw.real), preprocess(raw.imag)
        return real + 1j * imag
