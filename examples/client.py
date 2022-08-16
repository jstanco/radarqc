import argparse
import socket
import sys


def getargs() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", type=str, help="Location of cross-spectrum file"
    )
    parser.add_argument(
        "-a",
        "--address",
        type=str,
        help="Host IPv4 address",
        default="127.0.0.1",
    )
    parser.add_argument(
        "-p", "--port", type=int, help="Host port", required=True
    )
    return parser.parse_args()


def client(config: argparse.Namespace) -> int:
    with open(config.path, "rb") as f:
        raw = f.read()

    address = (config.address, config.port)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect(address)
        sockfile = sock.makefile(mode="wrb")
        sockfile.write(raw)
        return int(sockfile.read(1).decode())


def main() -> int:
    config = getargs()
    return client(config)


if __name__ == "__main__":
    sys.exit(main())
