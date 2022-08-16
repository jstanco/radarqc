import argparse

import matplotlib.pyplot as plt
import numpy as np

from radarqc import csfile


def getargs() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", type=str, help="Location of cross-spectrum file"
    )
    return parser.parse_args()


def plot_spectrum(cs: csfile.CSFile) -> None:
    assert cs.header.version >= 4

    ranges = cs.header.range_cell_dist_km * np.arange(
        0, cs.header.num_range_cells
    )

    freqs = (
        cs.header.rep_freq_mhz
        + 0.001
        * cs.header.bandwidth_khz
        * np.linspace(-0.5, 0.5, num=cs.header.num_doppler_cells, endpoint=True)
    )

    spectra = cs.antenna1, cs.antenna2, cs.antenna3

    # Plot each slice as an independent subplot
    fig, axes = plt.subplots(nrows=len(spectra), ncols=1)
    for i, (spectrum, ax) in enumerate(zip(spectra, axes.flat)):
        ax.pcolor(freqs, ranges, 10 * np.log10(spectrum))
        ax.set_ylabel("Range [km]")
        ax.set_xlabel("Frequency [MHz]")
        ax.set_title(
            "Range-Dependent Power Spectral Density (Antenna {})".format(i)
        )

    fig.tight_layout()
    plt.show()


def main():
    config = getargs()
    with open(config.path, "rb") as f:
        plot_spectrum(cs=csfile.load(f))


if __name__ == "__main__":
    main()
