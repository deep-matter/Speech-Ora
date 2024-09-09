import torch
import logging
from threading import Thread
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    TextIteratorStreamer
)
from src.sTs.Handlers import BaseHandler  
from nltk import sent_tokenize
from .chat import Chat
# Initialize logger
logger = logging.getLogger(__name__)

class LanguageModelHandler(BaseHandler):
    """
    Handles the language model part. 
    """

    def setup(
        self,
        model_name="TinyLlama/TinyLlama_v1.1",
        device="cpu", 
        torch_dtype="float16",
        gen_kwargs={},
        user_role="user",
        chat_size=3,
        init_chat_role=None, 
        init_chat_prompt="You are a helpful AI assistant.",
    ):
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)

        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
            #device_map="auto"

        ).to(device)
        
        # Create text-generation pipeline
        self.pipe = pipeline(
            "text-generation", 
            model=self.model, 
            tokenizer=self.tokenizer
        )

        # Initialize streamer for live streaming of generated text
        self.streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        self.gen_kwargs = {
            "streamer": self.streamer,
            "return_full_text": False,
            **gen_kwargs
        }

        # Initialize chat structure
        self.chat = Chat(chat_size)
        if init_chat_role:
            if not init_chat_prompt:
                raise ValueError(f"An initial prompt needs to be specified when setting init_chat_role.")
            self.chat.init_chat(
                {"role": init_chat_role, "content": init_chat_prompt}
            )
        self.user_role = user_role

        # Perform warmup
        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")

        # Dummy input for warmup
        dummy_input_text = "Write me a poem about Machine Learning."
        dummy_chat = [{"role": self.user_role, "content": dummy_input_text}]
        warmup_gen_kwargs = {
            "min_new_tokens": self.gen_kwargs.get("max_new_tokens", 50),
            "max_new_tokens": self.gen_kwargs.get("max_new_tokens", 50),
            **self.gen_kwargs
        }

        # Measure warmup time
        n_steps = 2
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_event.record()
        for _ in range(n_steps):
            thread = Thread(target=self.pipe, args=(dummy_chat,), kwargs=warmup_gen_kwargs)
            thread.start()
            for _ in self.streamer: 
                pass    
        end_event.record()
        torch.cuda.synchronize()

        logger.info(f"{self.__class__.__name__} warmed up! Time: {start_event.elapsed_time(end_event) * 1e-3:.3f} s")

    def process(self, prompt):
        logger.debug("Inferring language model...")

        # Add user prompt to chat
        self.chat.append(
            {"role": self.user_role, "content": prompt}
        )

        # Start text generation in a separate thread
        thread = Thread(target=self.pipe, args=(self.chat.to_list(),), kwargs=self.gen_kwargs)
        thread.start()

        generated_text, printable_text = "", ""
        for new_text in self.streamer:
            generated_text += new_text
            printable_text += new_text
            sentences = sent_tokenize(printable_text)
            
            # Yield sentence once it's complete
            if len(sentences) > 1:
                yield sentences[0]
                printable_text = new_text

        # Add the final generated content to the chat
        self.chat.append(
            {"role": "assistant", "content": generated_text}
        )

        # Yield the remaining sentence
        yield printable_text
