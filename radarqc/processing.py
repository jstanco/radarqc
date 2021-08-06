import abc
import numpy as np


class SignalProcessor(abc.ABC):
    """Base class for representing a signal processor, used to process
    HF radar spectra"""

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        return self._process(signal)

    @abc.abstractmethod
    def _process(self, signal: np.ndarray) -> np.ndarray:
        """Subclasses will override this functionality"""


class GainCalculator(SignalProcessor):
    """Convert the signal from raw Voltages into dB, given some
    reference gain as a baseline"""

    def __init__(self, reference: float) -> None:
        self._reference = reference

    def _process(self, signal: np.ndarray) -> np.ndarray:
        return 10 * np.log(signal) - self._reference


class Rectifier(SignalProcessor):
    """Zeros out all negative parts of a signal.  This can be useful
    for dealing with negative values in the signal, which are added to
    indicate outliers in the raw voltage data"""

    def _process(self, signal: np.ndarray) -> np.ndarray:
        return signal.clip(min=0)


class Abs(SignalProcessor):
    """Calculates absolute value of a signal.  This can be useful
    for dealing with negative values in the signal, which are added to
    indicate outliers in the raw voltage data"""

    def _process(self, signal: np.ndarray) -> np.ndarray:
        return np.abs(signal)


class Normalize(SignalProcessor):
    """Affine scaling such that the minimum signal value is equal to 0, and the
    maximum value is equal to 1"""

    def _process(self, signal: np.ndarray) -> np.ndarray:
        return (signal - signal.min()) / (signal.max() - signal.min())


class CompositeProcessor(SignalProcessor):
    """Represents a  composition of multiple processors into a single
    processor, allowing for creation of custom processing pipelines"""

    def __init__(self, *processors) -> None:
        self._processors = [p for p in processors]

    def _process(self, signal: np.ndarray) -> np.ndarray:
        for process in self._processors:
            signal = process(signal)
        return signal


class Identity(SignalProcessor):
    """Does nothing, returns the input signal without copying"""

    def _process(self, signal: np.ndarray) -> np.ndarray:
        return signal
