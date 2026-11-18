"""TCP transport for the native acquisition protocol (not the SCPI service).

Extracted from this project's BenchmarkSCPI transport in ultrasonic_redpitaya.
The full third-party Red Pitaya SCPI command wrapper is not needed here.
"""
import socket
import sys


class NativeTransport:
    """Handle partial TCP reads and disconnected peers on the native server."""

    def __init__(self, host, timeout=5.0, port=5001):
        self.host, self.port, self.timeout = host, port, timeout
        self._socket = None
        self._socket = socket.create_connection((host, port), timeout)
        self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def close(self):
        """Close the connection; repeated calls are harmless."""
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def __del__(self):
        self.close()

    def enable_quick_ack(self):
        """Windows per-socket ACK frequency; does not modify registry settings."""
        if sys.platform != 'win32':
            raise RuntimeError('Quick ACK is currently implemented only for Windows')
        import ctypes
        from ctypes import wintypes
        winsock = ctypes.WinDLL('Ws2_32.dll')
        ioctl = winsock.WSAIoctl
        ioctl.argtypes = [ctypes.c_size_t, wintypes.DWORD, ctypes.c_void_p,
                          wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
                          ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p, ctypes.c_void_p]
        ioctl.restype = ctypes.c_int
        frequency = wintypes.DWORD(1)
        returned = wintypes.DWORD()
        if ioctl(self._socket.fileno(), 0x98000017, ctypes.byref(frequency),
                 ctypes.sizeof(frequency), None, 0, ctypes.byref(returned), None, None):
            raise OSError(winsock.WSAGetLastError(), 'SIO_TCP_SET_ACK_FREQUENCY failed')

    def read_exact(self, size):
        """Read a complete protocol field or fail on an early disconnect."""
        data = bytearray()
        while len(data) < size:
            chunk = self._socket.recv(size - len(data))
            if not chunk:
                raise ConnectionError('Native acquisition connection closed')
            data.extend(chunk)
        return data
