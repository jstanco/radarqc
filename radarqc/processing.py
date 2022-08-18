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
    """Convert the signal from voltage-squared into dBW, given an optional
    reference gain as a baseline.  The impedance argument specifies the
    impedance of the rf frontend"""

    def __init__(self, reference: float = 0, impedance: float = 50) -> None:
        self._reference = reference
        self._impedance = impedance

    def _process(self, signal: np.ndarray) -> np.ndarray:
        # The input already has units volts**2.  We produce watts by dividing
        # by the rf frontend impedance.
        return 10 * np.log10(signal / self._impedance) - self._reference


class Rectifier(SignalProcessor):
    """Zeros out all negative parts of a signal.  This can be useful
    for dealing with negative values in the signal, which are added to
    indicate outliers in the raw voltage**2 data"""

    def _process(self, signal: np.ndarray) -> np.ndarray:
        return signal.clip(min=0)


class Abs(SignalProcessor):
    """Calculates absolute value of a signal.  This can be useful
    for dealing with negative values in the signal, which are added to
    indicate outliers in the raw voltage**2 data"""

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

    def __init__(self, *processors: SignalProcessor) -> None:
        self._processors = processors

    def _process(self, signal: np.ndarray) -> np.ndarray:
        for process in self._processors:
            signal = process(signal)
        return signal


class Identity(SignalProcessor):
    """Does nothing, a.k.a. returns the input signal."""

    def __init__(self, copy: bool = False) -> None:
        self._copy = copy

    def _process(self, signal: np.ndarray) -> np.ndarray:
        return signal.copy() if self._copy else signal
