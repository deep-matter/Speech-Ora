import logging
import os
import sys
from copy import copy
from pathlib import Path
from threading import Event
import torch
import nltk
from queue import Queue
from transformers import HfArgumentParser
from rich.console import Console

from models.detect import VADHandler
from models.LM import LanguageModelHandler
from models.stt import WhisperSTTHandler
from models.tts import ParlerTTSHandler
from models.chat import Chat
from Threads import ThreadManager


from SpeechArgument import (
    ModuleArguments,
    SocketReceiverArguments,
    SocketSenderArguments,
    ParlerTTSHandlerArguments,
    VADHandlerArguments,
    WhisperSTTHandlerArguments,
    LanguageModelHandlerArguments
)

from Handlers import (
    BaseHandler,
    SocketReceiver,
    SocketSender
)


# Ensure that the necessary NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Caching for compilation time reduction
CURRENT_DIR = Path(__file__).resolve().parent
os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(CURRENT_DIR, "tmp")
torch._inductor.config.fx_graph_cache = True
torch._dynamo.config.cache_size_limit = 15

console = Console()

def prepare_args(args, prefix):
    """
    Rename arguments by removing the prefix and prepares the gen_kwargs.
    """
    gen_kwargs = {}
    for key in copy(args.__dict__):
        if key.startswith(prefix):
            value = args.__dict__.pop(key)
            new_key = key[len(prefix) + 1:]  # Remove prefix and underscore
            if new_key.startswith("gen_"):
                gen_kwargs[new_key[4:]] = value  # Remove 'gen_' and add to dict
            else:
                args.__dict__[new_key] = value

    args.__dict__["gen_kwargs"] = gen_kwargs

def main():
    parser = HfArgumentParser((
        ModuleArguments,
        SocketReceiverArguments,
        SocketSenderArguments,
        VADHandlerArguments,
        WhisperSTTHandlerArguments,
        LanguageModelHandlerArguments,
        ParlerTTSHandlerArguments,
    ))

    # 0. Parse CLI arguments
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # Parse configurations from a JSON file if specified
        (
            module_kwargs,
            socket_receiver_kwargs,
            socket_sender_kwargs,
            vad_handler_kwargs,
            whisper_stt_handler_kwargs,
            language_model_handler_kwargs,
            parler_tts_handler_kwargs,
        ) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # Parse arguments from command line if no JSON file is provided
        (
            module_kwargs,
            socket_receiver_kwargs,
            socket_sender_kwargs,
            vad_handler_kwargs,
            whisper_stt_handler_kwargs,
            language_model_handler_kwargs,
            parler_tts_handler_kwargs,
        ) = parser.parse_args_into_dataclasses()

    # 1. Handle logger
    global logger
    logging.basicConfig(
        level=module_kwargs.log_level.upper(),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    logger = logging.getLogger(__name__)

    # torch compile logs
    if module_kwargs.log_level == "debug":
        torch._logging.set_logs(graph_breaks=True, recompiles=True, cudagraphs=True)

    # 2. Prepare each part's arguments
    prepare_args(whisper_stt_handler_kwargs, "stt")
    prepare_args(language_model_handler_kwargs, "lm")
    prepare_args(parler_tts_handler_kwargs, "tts")

    # 3. Build the pipeline
    stop_event = Event()
    should_listen = Event()
    recv_audio_chunks_queue = Queue()
    send_audio_chunks_queue = Queue()
    spoken_prompt_queue = Queue()
    text_prompt_queue = Queue()
    lm_response_queue = Queue()

    vad = VADHandler(
        stop_event,
        queue_in=recv_audio_chunks_queue,
        queue_out=spoken_prompt_queue,
        setup_args=(should_listen,),
        setup_kwargs=vars(vad_handler_kwargs),
    )
    stt = WhisperSTTHandler(
        stop_event,
        queue_in=spoken_prompt_queue,
        queue_out=text_prompt_queue,
        setup_kwargs=vars(whisper_stt_handler_kwargs),
    )
    lm = LanguageModelHandler(
        stop_event,
        queue_in=text_prompt_queue,
        queue_out=lm_response_queue,
        setup_kwargs=vars(language_model_handler_kwargs),
    )
    tts = ParlerTTSHandler(
        stop_event,
        queue_in=lm_response_queue,
        queue_out=send_audio_chunks_queue,
        setup_args=(should_listen,),
        setup_kwargs=vars(parler_tts_handler_kwargs),
    )

    recv_handler = SocketReceiver(
        stop_event,
        recv_audio_chunks_queue,
        should_listen,
        host=socket_receiver_kwargs.recv_host,
        port=socket_receiver_kwargs.recv_port,
        chunk_size=socket_receiver_kwargs.chunk_size,
    )

    send_handler = SocketSender(
        stop_event,
        send_audio_chunks_queue,
        host=socket_sender_kwargs.send_host,
        port=socket_sender_kwargs.send_port,
    )

    # 4. Run the pipeline
    try:
        pipeline_manager = ThreadManager([vad, tts, lm, stt, recv_handler, send_handler])
        pipeline_manager.start()

    except KeyboardInterrupt:
        pipeline_manager.stop()

if __name__ == "__main__":
    main()
