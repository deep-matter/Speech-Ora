from dataclasses import dataclass, field

@dataclass
class ListenAndPlayArguments:
    """
    Arguments for the ListenAndPlay class.

    Attributes:
        send_rate (int): The sample rate for sending audio data (default: 16000 Hz).
        recv_rate (int): The sample rate for receiving audio data (default: 44100 Hz).
        list_play_chunk_size (int): The size of data chunks (in bytes) for audio processing (default: 1024).
        host (str): The hostname or IP address for listening and playing (default: 'localhost').
        send_port (int): The network port for sending data (default: 12345).
        recv_port (int): The network port for receiving data (default: 12346).
    """
    send_rate: int = field(
        default=16000,
        metadata={
            "help": "In Hz. Default is 16000."
        }
    )
    recv_rate: int = field(
        default=44100,
        metadata={
            "help": "In Hz. Default is 44100."
        }
    )
    list_play_chunk_size: int = field(
        default=1024,
        metadata={
            "help": "The size of data chunks (in bytes). Default is 1024."
        }
    )
    host: str = field(
        default="localhost",
        metadata={
            "help": "The hostname or IP address for listening and playing. Default is 'localhost'."
        }
    )
    send_port: int = field(
        default=12345,
        metadata={
            "help": "The network port for sending data. Default is 12345."
        }
    )
    recv_port: int = field(
        default=12346,
        metadata={
            "help": "The network port for receiving data. Default is 12346."
        }
    )
