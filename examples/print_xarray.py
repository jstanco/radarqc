import argparse

from radarqc import csfile


def getargs() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", type=str, help="Location of cross-spectrum file"
    )
    return parser.parse_args()


def main():
    config = getargs()
    with open(config.path, "rb") as f:
        print(csfile.load(f).to_xarray())


if __name__ == "__main__":
    main()
