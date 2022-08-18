import argparse

import matplotlib.pyplot as plt
import xarray as xr

from radarqc import csfile
from radarqc.processing import Abs, CompositeProcessor, GainCalculator


def getargs() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", type=str, help="Location of cross-spectrum file"
    )
    return parser.parse_args()


def plot_spectrum(ds: xr.Dataset) -> None:
    spectra = ds.antenna1, ds.antenna2, ds.antenna3

    # Plot each slice as an independent subplot
    fig, axes = plt.subplots(nrows=len(spectra), ncols=1)
    for i, (spectrum, ax) in enumerate(zip(spectra, axes.flat)):
        ax.pcolor(ds.radiation_frequency, ds.range, spectrum)
        ax.set_ylabel("{} [{}]".format(ds.range.name, ds.range.units))
        ax.set_xlabel(
            "{} [{}]".format(
                ds.radiation_frequency.name, ds.radiation_frequency.units
            )
        )
        ax.set_title(
            "Range-Dependent Power Spectral Density (Antenna {})".format(i + 1)
        )

    fig.tight_layout()
    plt.show()


def main():
    config = getargs()
    with open(config.path, "rb") as f:
        preprocess = CompositeProcessor(Abs(), GainCalculator())
        plot_spectrum(csfile.load(f, preprocess=preprocess).to_xarray())


if __name__ == "__main__":
    main()
