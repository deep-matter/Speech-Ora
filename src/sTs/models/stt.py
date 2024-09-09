import torch
import logging
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor
)
from src.sTs.Handlers import BaseHandler  # Assuming this exists in your src directory

# Initialize logger
logger = logging.getLogger(__name__)

class WhisperSTTHandler(BaseHandler):
    """
    Handles the Speech-To-Text generation using a Whisper model.
    """

    def setup(
        self,
        model_name="distil-whisper/distil-large-v3",
        device="cpu",  
        torch_dtype="float16",  
        compile_mode=None,
        gen_kwargs={}
    ): 
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)
        self.compile_mode = compile_mode
        self.gen_kwargs = gen_kwargs

        # Initialize processor and model
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
            offload_state_dict=False,
            #device_map="auto"

        ).to(device)
        
        # Compile model if compile mode is specified
        if self.compile_mode:
            self.model.generation_config.cache_implementation = "static"
            self.model.forward = torch.compile(self.model.forward, mode=self.compile_mode, fullgraph=True)
        
        self.warmup()

    def prepare_model_inputs(self, spoken_prompt):
        """
        Prepares model inputs by processing the spoken prompt.
        """
        input_features = self.processor(
            spoken_prompt, sampling_rate=16000, return_tensors="pt"
        ).input_features
        input_features = input_features.to(self.device, dtype=self.torch_dtype)

        return input_features
        
    def warmup(self):
        """
        Performs warmup for the model to ensure efficient runtime performance.
        """
        logger.info(f"Warming up {self.__class__.__name__}")

        # Determine number of warmup steps
        n_steps = 1 if self.compile_mode == "default" else 2
        dummy_input = torch.randn(
            (1, self.model.config.num_mel_bins, 3000),
            dtype=self.torch_dtype,
            device=self.device
        )

        warmup_gen_kwargs = self.gen_kwargs

        if self.compile_mode not in (None, "default"):
            warmup_gen_kwargs = {
                "min_new_tokens": self.gen_kwargs.get("max_new_tokens", 50),
                "max_new_tokens": self.gen_kwargs.get("max_new_tokens", 50),
                **self.gen_kwargs
            }

        # Measure warmup timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_event.record()
        for _ in range(n_steps):
            _ = self.model.generate(dummy_input, **warmup_gen_kwargs)
        end_event.record()
        torch.cuda.synchronize()

        logger.info(f"{self.__class__.__name__} warmed up! Time: {start_event.elapsed_time(end_event) * 1e-3:.3f} s")

    def process(self, spoken_prompt):
        """
        Runs inference on the spoken input and generates corresponding text.
        """
        logger.debug("Inferring whisper...")

        # Prepare inputs for the model
        input_features = self.prepare_model_inputs(spoken_prompt)

        # Generate prediction IDs from the model
        pred_ids = self.model.generate(input_features, **self.gen_kwargs)

        # Decode prediction to get the text output
        pred_text = self.processor.batch_decode(
            pred_ids, 
            skip_special_tokens=True,
            decode_with_timestamps=False
        )[0]

        logger.debug("Finished whisper inference")
        print(f"USER: {pred_text}")

        yield pred_text
