import argparse
import socket
import sys

from typing import Callable

from radarqc import csfile, CSFile


def getargs() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a", "--address", type=str, help="Host IPv4 address", default="0.0.0.0"
    )
    parser.add_argument(
        "-p", "--port", type=int, help="Host port", required=True
    )
    return parser.parse_args()


class Server:
    def __init__(self, config: argparse.Namespace) -> None:
        self._callbacks = []
        self._address = (config.address, config.port)

    def register_callback(self, callback: Callable[[CSFile], None]):
        self._callbacks.append(callback)
        return self

    def run(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(self._address)
            sock.listen()

            while True:
                conn, _ = sock.accept()
                sockfile = conn.makefile(mode="rwb")
                cs = csfile.load(sockfile)
                for callback in self._callbacks:
                    callback(cs)
                sockfile.write(str(0).encode())
                sockfile.flush()
                conn.close()


def print_header(cs: CSFile) -> None:
    print(cs.header)


def main() -> None:
    Server(config=getargs()).register_callback(callback=print_header).run()


if __name__ == "__main__":
    sys.exit(main())
