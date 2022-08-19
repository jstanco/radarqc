import argparse

from radarqc import csfile


def getargs() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_path", type=str, help="Location of input cross-spectrum file"
    )
    parser.add_argument(
        "output_path", type=str, help="Location of output netcdf file"
    )
    return parser.parse_args()


def main():
    config = getargs()
    with open(config.input_path, "rb") as f:
        csfile.load(f).to_xarray().to_netcdf(config.output_path)


if __name__ == "__main__":
    main()
