from dataclasses import dataclass, field

@dataclass
class ModuleArguments:
    log_level: str = field(
        default="info",
        metadata={
            "help": "Provide logging level. Example --log_level debug, default=warning."
        }
    )

@dataclass
class SocketReceiverArguments:
    recv_host: str = field(
        default="localhost",
        metadata={
            "help": "The host IP ddress for the socket connection. Default is '0.0.0.0' which binds to all "
                    "available interfaces on the host machine."
        }
    )
    recv_port: int = field(
        default=12345,
        metadata={
            "help": "The port number on which the socket server listens. Default is 12346."
        }
    )
    chunk_size: int = field(
        default=1024,
        metadata={
            "help": "The size of each data chunk to be sent or received over the socket. Default is 1024 bytes."
        }
    )


@dataclass
class SocketSenderArguments:
    send_host: str = field(
        default="localhost",
        metadata={
            "help": "The host IP address for the socket connection. Default is '0.0.0.0' which binds to all "
                    "available interfaces on the host machine."
        }
    )
    send_port: int = field(
        default=12346,
        metadata={
            "help": "The port number on which the socket server listens. Default is 12346."
        }
    )

@dataclass
class VADHandlerArguments:
    thresh: float = field(
        default=0.3,
        metadata={
            "help": "The threshold value for voice activity detection (VAD). Values typically range from 0 to 1, with higher values requiring higher confidence in speech detection."
        }
    )
    sample_rate: int = field(
        default=16000,
        metadata={
            "help": "The sample rate of the audio in Hertz. Default is 16000 Hz, which is a common setting for voice audio."
        }
    )
    min_silence_ms: int = field(
        default=250,
        metadata={
            "help": "Minimum length of silence intervals to be used for segmenting speech. Measured in milliseconds. Default is 1000 ms."
        }
    )
    min_speech_ms: int = field(
        default=500,
        metadata={
            "help": "Minimum length of speech segments to be considered valid speech. Measured in milliseconds. Default is 500 ms."
        }
    )
    max_speech_ms: float = field(
        default=float('inf'),
        metadata={
            "help": "Maximum length of continuous speech before forcing a split. Default is infinite, allowing for uninterrupted speech segments."
        }
    )
    speech_pad_ms: int = field(
        default=30,
        metadata={
            "help": "Amount of padding added to the beginning and end of detected speech segments. Measured in milliseconds. Default is 30 ms."
        }
    )

@dataclass
class WhisperSTTHandlerArguments:
    stt_model_name: str = field(
        default="distil-whisper/distil-large-v3",
        metadata={
            "help": "The pretrained Whisper model to use. Default is 'distil-whisper/distil-large-v3'."
        }
    )
    stt_device: str = field(
        default="cuda",
        metadata={
            "help": "The device type on which the model will run. Default is 'cuda' for GPU acceleration."
        }
    )
    stt_torch_dtype: str = field(
        default="float16",
        metadata={
            "help": "The PyTorch data type for the model and input tensors. One of `float32` (full-precision), `float16` or `bfloat16` (both half-precision)."
        } 
    )
    stt_compile_mode: str = field(
        default=None,
        metadata={
            "help": "Compile mode for torch compile. Either 'default', 'reduce-overhead' and 'max-autotune'. Default is None (no compilation)"
        }
    )
    stt_gen_max_new_tokens: int = field(
        default=128,
        metadata={
            "help": "The maximum number of new tokens to generate. Default is 128."
        }
    )
    stt_gen_num_beams: int = field(
        default=1,
        metadata={
            "help": "The number of beams for beam search. Default is 1, implying greedy decoding."
        }
    )
    stt_gen_return_timestamps: bool = field(
        default=False,
        metadata={
            "help": "Whether to return timestamps with transcriptions. Default is False."
        }
    )
    stt_gen_task: str = field(
        default="transcribe",
        metadata={
            "help": "The task to perform, typically 'transcribe' for transcription. Default is 'transcribe'."
        }
    )
    stt_gen_language: str = field(
        default="en",
        metadata={
            "help": "The language of the speech to transcribe. Default is 'en' for English."
        }
    )

@dataclass
class LanguageModelHandlerArguments:
    lm_model_name: str = field(
        default="TinyLlama/TinyLlama_v1.1",
        metadata={
            "help": "The pretrained language model to use. Default is 'microsoft/Phi-3-mini-4k-instruct'."
        }
    )
    lm_device: str = field(
        default="cuda",
        metadata={
            "help": "The device type on which the model will run. Default is 'cuda' for GPU acceleration."
        }
    )
    lm_torch_dtype: str = field(
        default="float16",
        metadata={
            "help": "The PyTorch data type for the model and input tensors. One of `float32` (full-precision), `float16` or `bfloat16` (both half-precision)."
        }
    )
    user_role: str = field(
        default="user",
        metadata={
            "help": "Role assigned to the user in the chat context. Default is 'user'."
        }
    )
    init_chat_role: str = field(
        default=None,
        metadata={
            "help": "Initial role for setting up the chat context. Default is 'system'."
        }
    )
    init_chat_prompt: str = field(
        default="You are a helpful AI assistant.",
        metadata={
            "help": "The initial chat prompt to establish context for the language model. Default is 'You are a helpful AI assistant.'"
        }
    )
    lm_gen_max_new_tokens: int = field(
        default=64,
        metadata={"help": "Maximum number of new tokens to generate in a single completion. Default is 128."}
    )
    lm_gen_temperature: float = field(
        default=0.0,
        metadata={"help": "Controls the randomness of the output. Set to 0.0 for deterministic (repeatable) outputs. Default is 0.0."}
    )
    lm_gen_do_sample: bool = field(
        default=False,
        metadata={"help": "Whether to use sampling; set this to False for deterministic outputs. Default is False."}
    )
    chat_size: int = field(
        default=3,
        metadata={"help": "Number of messages of the messages to keep for the chat. None for no limitations."}
    )
    
@dataclass
class ParlerTTSHandlerArguments:
    tts_model_name: str = field(
        default="ylacombe/parler-tts-mini-jenny-30H",
        metadata={
            "help": "The pretrained TTS model to use. Default is 'ylacombe/parler-tts-mini-jenny-30H'."
        }
    )
    tts_device: str = field(
        default="cuda",
        metadata={
            "help": "The device type on which the model will run. Default is 'cuda' for GPU acceleration."
        }
    )
    tts_torch_dtype: str = field(
        default="float16",
        metadata={
            "help": "The PyTorch data type for the model and input tensors. One of `float32` (full-precision), `float16` or `bfloat16` (both half-precision)."
        }
    )
    tts_compile_mode: str = field(
        default=None,
        metadata={
            "help": "Compile mode for torch compile. Either 'default', 'reduce-overhead' and 'max-autotune'. Default is None (no compilation)"
        }
    )
    tts_gen_min_new_tokens: int = field(
        default=None,
        metadata={"help": "Maximum number of new tokens to generate in a single completion. Default is 10, which corresponds to ~0.1 secs"}
    )
    tts_gen_max_new_tokens: int = field(
        default=512,
        metadata={"help": "Maximum number of new tokens to generate in a single completion. Default is 256, which corresponds to ~6 secs"}
    )
    description: str = field(
        default=(
            "A female speaker with a slightly low-pitched voice delivers her words quite expressively, in a very confined sounding environment with clear audio quality. "
            "She speaks very fast."
        ),
        metadata={
            "help": "Description of the speaker's voice and speaking style to guide the TTS model."
        }
    )
    play_steps_s: float = field(
        default=0.2,
        metadata={
            "help": "The time interval in seconds for playing back the generated speech in steps. Default is 0.5 seconds."
        }
    )
    max_prompt_pad_length: int = field(
        default=8,
        metadata={
            "help": "When using compilation, the prompt as to be padded to closest power of 2. This parameters sets the maximun power of 2 possible."
        }
    ) 
