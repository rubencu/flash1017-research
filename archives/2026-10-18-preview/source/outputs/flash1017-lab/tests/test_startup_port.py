import socket

import pytest

from flash1017.serve import check_port_available


def listener():
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('127.0.0.1', 0))
    server.listen(1)
    return server


def test_a_live_listener_is_still_rejected():
    with listener() as server:
        with pytest.raises(RuntimeError, match='already in use'):
            check_port_available(server.getsockname()[1])


def test_recently_closed_connection_does_not_block_restart():
    with listener() as server:
        address = server.getsockname()
        with socket.create_connection(address) as client:
            with server.accept()[0] as accepted:
                # Server initiates FIN, leaving its endpoint in TIME_WAIT.
                accepted.shutdown(socket.SHUT_WR)
                assert client.recv(1) == b''
    with socket.socket() as old_check:
        with pytest.raises(OSError):
            old_check.bind(address)
    check_port_available(address[1])
