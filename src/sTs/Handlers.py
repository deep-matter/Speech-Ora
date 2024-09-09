import socket
import logging
from time import perf_counter

# Initialize logger
logger = logging.getLogger(__name__)

class BaseHandler:
    """
    Base class for pipeline parts. Each part of the pipeline has an input and an output queue.
    The `setup` method along with `setup_args` and `setup_kwargs` can be used to address the specific requirements of the implemented pipeline part.
    To stop a handler properly, set the stop_event and, to avoid queue deadlocks, place b"END" in the input queue.
    Objects placed in the input queue will be processed by the `process` method, and the yielded results will be placed in the output queue.
    The cleanup method handles stopping the handler, and b"END" is placed in the output queue.
    """

    def __init__(self, stop_event, queue_in, queue_out, setup_args=(), setup_kwargs={}):
        self.stop_event = stop_event
        self.queue_in = queue_in
        self.queue_out = queue_out
        self.setup(*setup_args, **setup_kwargs)
        self._times = []

    def setup(self):
        pass

    def process(self, input_data):
        raise NotImplementedError

    def run(self):
        while not self.stop_event.is_set():
            input_data = self.queue_in.get()
            if isinstance(input_data, bytes) and input_data == b'END':
                # Sentinel signal to avoid queue deadlock
                logger.debug("Stopping thread")
                break
            start_time = perf_counter()
            for output in self.process(input_data):
                self._times.append(perf_counter() - start_time)
                logger.debug(f"{self.__class__.__name__}: {self.last_time:.3f} s")
                self.queue_out.put(output)
                start_time = perf_counter()

        self.cleanup()
        self.queue_out.put(b'END')

    @property
    def last_time(self):
        return self._times[-1]

    def cleanup(self):
        pass


class SocketReceiver:
    """
    Handles reception of the audio packets from the client.
    """

    def __init__(
        self, 
        stop_event,
        queue_out,
        should_listen,
        host='0.0.0.0', 
        port=12345,
        chunk_size=1024
    ):
        self.stop_event = stop_event
        self.queue_out = queue_out
        self.should_listen = should_listen
        self.chunk_size = chunk_size
        self.host = host
        self.port = port

    def receive_full_chunk(self, conn, chunk_size):
        data = b''
        while len(data) < chunk_size:
            packet = conn.recv(chunk_size - len(data))
            if not packet:
                return None  # Connection closed
            data += packet
        return data

    def run(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)
        logger.info('Receiver waiting to be connected...')
        self.conn, _ = self.socket.accept()
        logger.info("Receiver connected")

        self.should_listen.set()
        while not self.stop_event.is_set():
            audio_chunk = self.receive_full_chunk(self.conn, self.chunk_size)
            if audio_chunk is None:
                self.queue_out.put(b'END')
                break
            if self.should_listen.is_set():
                self.queue_out.put(audio_chunk)
        self.conn.close()
        logger.info("Receiver closed")


class SocketSender:
    """
    Handles sending generated audio packets to the client.
    """

    def __init__(
        self, 
        stop_event,
        queue_in,
        host='0.0.0.0', 
        port=12346
    ):
        self.stop_event = stop_event
        self.queue_in = queue_in
        self.host = host
        self.port = port

    def run(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)
        logger.info('Sender waiting to be connected...')
        self.conn, _ = self.socket.accept()
        logger.info("Sender connected")

        while not self.stop_event.is_set():
            audio_chunk = self.queue_in.get()
            self.conn.sendall(audio_chunk)
            if isinstance(audio_chunk, bytes) and audio_chunk == b'END':
                break
        self.conn.close()
        logger.info("Sender closed")
