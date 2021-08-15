import argparse
import socket
import sys

from radarqc import csfile


def getargs() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a", "--address", type=str, help="Host IPv4 address", default="0.0.0.0"
    )
    parser.add_argument("-p", "--port", type=int, help="Host port")
    return parser.parse_args()


def server(config: argparse.Namespace) -> None:
    address = (config.address, config.port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(address)
    sock.listen()

    while True:
        conn, _ = sock.accept()
        sockfile = conn.makefile(mode="rb")
        cs = csfile.load(sockfile)
        print(cs.header)


def main() -> None:
    config = getargs()
    server(config)


if __name__ == "__main__":
    sys.exit(main())
