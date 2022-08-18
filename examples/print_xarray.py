import argparse

import matplotlib.pyplot as plt

from radarqc import csfile
from radarqc.processing import GainCalculator


def getargs() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", type=str, help="Location of cross-spectrum file"
    )
    return parser.parse_args()


def main():
    config = getargs()
    with open(config.path, "rb") as f:
        ds = csfile.load(f, preprocess=GainCalculator()).to_xarray()
        print(ds)


if __name__ == "__main__":
    main()
