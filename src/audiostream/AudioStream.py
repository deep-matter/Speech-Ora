import socket
from queue import Queue
import threading
from typing import Tuple
import numpy as np
import sounddevice as sd
from transformers import HfArgumentParser
from Argument import ListenAndPlayArguments


class AudioStreamer:
    """Handles sending and receiving audio data over a network."""
    def __init__(self, send_rate: int, recv_rate: int, chunk_size: int, host: str, send_port: int, recv_port: int):
        """
        Initializes the AudioStreamer object.

        Attributes:
            send_rate (int): The sample rate for sending audio data.
            recv_rate (int): The sample rate for receiving audio data.
            chunk_size (int): The size of data chunks (in bytes) for audio processing.
            host (str): The hostname or IP address for listening and playing.
            send_port (int): The network port for sending data.
            recv_port (int): The network port for receiving data.
            send_socket (socket.socket): The network socket for sending data.
            recv_socket (socket.socket): The network socket for receiving data.
            stop_event (threading.Event): The event to stop the streaming.
            recv_queue (Queue): The queue to hold the received audio data.
            send_queue (Queue): The queue to hold the audio data to be sent.
        """
        self.send_rate = send_rate
        self.recv_rate = recv_rate
        self.chunk_size = chunk_size
        self.host = host
        self.send_port = send_port
        self.recv_port = recv_port
        self.send_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.recv_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.stop_event = threading.Event()
        self.recv_queue = Queue()
        self.send_queue = Queue()

    def connect(self) -> None:
        """Connects to the send and receive sockets."""
        self.send_socket.connect((self.host, self.send_port))
        self.recv_socket.connect((self.host, self.recv_port))
        print("Connected to send and receive sockets.")

    def _send(self) -> None:
        """Continuously sends audio data from the queue to the network socket."""
        while not self.stop_event.is_set():
            data = self.send_queue.get()
            self.send_socket.sendall(data)

    def _recv(self) -> None:
        """Continuously receives audio data from the network socket."""
        while not self.stop_event.is_set():
            data = self._receive_full_chunk(self.recv_socket, self.chunk_size * 2)
            if data:
                self.recv_queue.put(data)

    def _receive_full_chunk(self, conn: socket.socket, chunk_size: int) -> bytes:
        """Receives a complete chunk of data from the socket."""
        data = b''
        while len(data) < chunk_size:
            packet = conn.recv(chunk_size - len(data))
            if not packet:
                return None  # Connection has been closed
            data += packet
        return data


class AudioDevice:
    """Manages the audio device for sending and receiving audio."""
    def __init__(self, send_rate: int, recv_rate: int, chunk_size: int, send_queue: Queue, recv_queue: Queue):
        self.send_rate = send_rate
        self.recv_rate = recv_rate
        self.chunk_size = chunk_size
        self.send_queue = send_queue
        self.recv_queue = recv_queue

    def callback_recv(self, outdata: np.ndarray, frames: int, time: Tuple, status: sd.CallbackFlags) -> None:
        """Callback for receiving audio data and playing it through speakers."""
        if not self.recv_queue.empty():
            data = self.recv_queue.get()
            outdata[:len(data)] = data
            outdata[len(data):] = b'\x00' * (len(outdata) - len(data))
        else:
            outdata[:] = b'\x00' * len(outdata)

    def callback_send(self, indata: np.ndarray, frames: int, time: Tuple, status: sd.CallbackFlags) -> None:
        """Callback for sending audio data recorded from the microphone."""
        if self.recv_queue.empty():
            data = bytes(indata)
            self.send_queue.put(data)

    def start_device_streams(self) -> Tuple[sd.RawInputStream, sd.RawOutputStream]:
        """Starts the input and output audio streams."""
        send_stream = sd.RawInputStream(
            samplerate=self.send_rate, channels=1, dtype='int16', blocksize=self.chunk_size, callback=self.callback_send
        )
        recv_stream = sd.RawOutputStream(
            samplerate=self.recv_rate, channels=1, dtype='int16', blocksize=self.chunk_size, callback=self.callback_recv
        )
        return send_stream, recv_stream


def listen_and_play(
    send_rate: int,
    recv_rate: int,
    list_play_chunk_size: int,
    host: str,
    send_port: int,
    recv_port: int
) -> None:
    """Main function to handle audio listening and playing between two network sockets."""
    streamer = AudioStreamer(send_rate, recv_rate, list_play_chunk_size, host, send_port, recv_port)
    device = AudioDevice(send_rate, recv_rate, list_play_chunk_size, streamer.send_queue, streamer.recv_queue)

    streamer.connect()
    send_stream, recv_stream = device.start_device_streams()

    try:
        threading.Thread(target=send_stream.start).start()
        threading.Thread(target=recv_stream.start).start()

        streamer.start_streaming()
        input("Press Enter to stop...")

    except KeyboardInterrupt:
        print("Stopping streaming...")

    finally:
        streamer.stop_streaming()
        send_stream.stop()
        recv_stream.stop()

if __name__ == "__main__":
    parser = HfArgumentParser((ListenAndPlayArguments,))
    listen_and_play_kwargs, = parser.parse_args_into_dataclasses()
    listen_and_play(**vars(listen_and_play_kwargs))


