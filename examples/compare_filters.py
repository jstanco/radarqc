import glob
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from radarqc.dataset import DataSet
from radarqc.filtering import (
    NoiseFilter,
    PCAFilter,
    PreFitPCAFilter,
    SpectrumFilter,
)

from radarqc.processing import (
    Abs,
    CompositeProcessor,
    GainCalculator,
    Normalize,
    SignalProcessor,
)

from radarqc import csfile
from typing import Iterable


def plot_vertical(*images) -> None:
    image = np.concatenate(images)
    plt.imshow(image, aspect=4)
    plt.colorbar()
    plt.show()


def _time(fn, *args, **kwargs) -> tuple:
    t = time.time()
    result = fn(*args, **kwargs)
    return result, time.time() - t


def log_timing(t: float, name: str) -> None:
    line_length = 80
    print("-" * line_length)
    print("Time taken for {}: {} sec".format(name, t))
    print("-" * line_length)


def create_preprocessor() -> SignalProcessor:
    reference_gain = 34.2
    return CompositeProcessor(
        Abs(), GainCalculator(reference=reference_gain), Normalize()
    )


def compare_filters(
    spectrum: np.ndarray, filters: Iterable[SpectrumFilter]
) -> None:
    for filt in filters:
        name = filt.__class__.__name__
        filtered, t = _time(filt, spectrum)
        log_timing(t, name)
        plot_vertical(spectrum, filtered)


def generate_paths(dir: str) -> Iterable[str]:
    return glob.glob(os.path.join(dir, "*.cs"))


def main():
    base = "../../codar"
    paths = generate_paths(base)
    preprocess = create_preprocessor()

    dataset = DataSet(paths, preprocess)
    num_components = 0.8

    noise = NoiseFilter(threshold=0.18, window_std=0.02)
    pca = PCAFilter(num_components=num_components)
    prefit = PreFitPCAFilter(dataset.spectra, num_components)

    path = "../../codar/CSS_ASSA_21_06_26_1400.cs"
    with open(path, "rb") as f:
        cs = csfile.load(f, preprocess)
        filters = [prefit, pca, noise]
        compare_filters(cs.antenna3, filters)


if __name__ == "__main__":
    main()
