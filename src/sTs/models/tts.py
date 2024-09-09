import torch
import logging
import numpy as np
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSStreamer
from src.sTs.Handlers import BaseHandler  # Assuming this exists in your src directory
from threading import Thread

# Initialize logger
logger = logging.getLogger(__name__)

class ParlerTTSHandler(BaseHandler):
    """
    Handles Text-to-Speech generation using the Parler TTS model.
    """

    def setup(
        self,
        should_listen,
        model_name="ylacombe/parler-tts-mini-jenny-30H",
        device_map="auto", 
        device="cpu",
        torch_dtype="float16",
        compile_mode=None,
        gen_kwargs={},
        max_prompt_pad_length=8,
        description=(
            "A female speaker with a slightly low-pitched voice delivers her words quite expressively, "
            "in a very confined sounding environment with clear audio quality. She speaks very fast."
        ),
        play_steps_s=1
    ):
        """
        Sets up the TTS model and initializes various configurations.
        """
        self.should_listen = should_listen
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)
        self.gen_kwargs = gen_kwargs
        self.compile_mode = compile_mode
        self.max_prompt_pad_length = max_prompt_pad_length
        self.description = description

        # Initialize tokenizers and model
        self.description_tokenizer = AutoTokenizer.from_pretrained(model_name) 
        self.prompt_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
            offload_state_dict=False,
            
        ).to(device)

        # Calculate play steps based on frame rate
        framerate = self.model.audio_encoder.config.frame_rate
        self.play_steps = int(framerate * play_steps_s)

        # Compile model if needed
        if self.compile_mode not in (None, "default"):
            logger.warning("Torch compilation modes that capture CUDA graphs are not yet compatible with the STT part. Reverting to 'default'")
            self.compile_mode = "default"

        if self.compile_mode:
            self.model.generation_config.cache_implementation = "static"
            self.model.forward = torch.compile(self.model.forward, mode=self.compile_mode, fullgraph=True)

        self.warmup()

    def prepare_model_inputs(self, prompt, max_length_prompt=50, pad=False):
        """
        Prepares inputs for the model by tokenizing both the description and prompt.
        """
        pad_args_prompt = {"padding": "max_length", "max_length": max_length_prompt} if pad else {}

        # Tokenize description and prompt
        tokenized_description = self.description_tokenizer(self.description, return_tensors="pt")
        input_ids = tokenized_description.input_ids.to(self.device)
        attention_mask = tokenized_description.attention_mask.to(self.device)

        tokenized_prompt = self.prompt_tokenizer(prompt, return_tensors="pt", **pad_args_prompt)
        prompt_input_ids = tokenized_prompt.input_ids.to(self.device)
        prompt_attention_mask = tokenized_prompt.attention_mask.to(self.device)

        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            **self.gen_kwargs
        }

        return gen_kwargs
    
    def warmup(self):
        """
        Performs warmup steps to prepare the model for efficient inference.
        """
        logger.info(f"Warming up {self.__class__.__name__}")

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Determine number of warmup steps
        n_steps = 1 if self.compile_mode == "default" else 2

        torch.cuda.synchronize()
        start_event.record()

        if self.compile_mode:
            pad_lengths = [2**i for i in range(2, self.max_prompt_pad_length)]
            for pad_length in pad_lengths[::-1]:
                model_kwargs = self.prepare_model_inputs(
                    "dummy prompt", 
                    max_length_prompt=pad_length,
                    pad=True
                )
                for _ in range(n_steps):
                    _ = self.model.generate(**model_kwargs)
                logger.info(f"Warmed up length {pad_length} tokens!")
        else:
            model_kwargs = self.prepare_model_inputs("dummy prompt")
            for _ in range(n_steps):
                _ = self.model.generate(**model_kwargs)
                
        end_event.record() 
        torch.cuda.synchronize()

        logger.info(f"{self.__class__.__name__}:  warmed up! time: {start_event.elapsed_time(end_event) * 1e-3:.3f} s")

    def process(self, llm_sentence):
        """
        Processes the LLM sentence by generating audio output.
        """
        logger.info(f"Processing sentence: {llm_sentence}")
        nb_tokens = len(self.prompt_tokenizer(llm_sentence).input_ids)

        # Adjust padding if compile mode is enabled
        pad_args = {}
        if self.compile_mode:
            pad_length = next_power_of_2(nb_tokens)
            logger.debug(f"Padding to {pad_length}")
            pad_args["pad"] = True
            pad_args["max_length_prompt"] = pad_length
    
        # Prepare inputs for TTS
        tts_gen_kwargs = self.prepare_model_inputs(llm_sentence, **pad_args)

        # Initialize streamer for audio output
        streamer = ParlerTTSStreamer(self.model, device=self.device, play_steps=self.play_steps)
        tts_gen_kwargs = {
            "streamer": streamer,
            **tts_gen_kwargs
        }

        torch.manual_seed(0)
        thread = Thread(target=self.model.generate, kwargs=tts_gen_kwargs)
        thread.start()

        # Yield audio chunks for real-time playback
        for i, audio_chunk in enumerate(streamer):
            if i == 0:
                logger.info(f"Time to first audio: {perf_counter() - pipeline_start:.3f}")
            audio_chunk = np.int16(audio_chunk * 32767)  # Convert to int16 format
            yield audio_chunk

        self.should_listen.set()
