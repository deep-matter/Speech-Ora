import torch
import numpy as np
import logging
import sys
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.helpers import VADIterator, int2float  # Import from the correct path
from src.sTs.Handlers import BaseHandler 
# Initialize logger
logger = logging.getLogger(__name__)


class VADHandler(BaseHandler):
    """
    Handles voice activity detection. When voice activity is detected, audio will be accumulated 
    until the end of speech is detected and then passed to the next stage.
    """

    def setup(
        self, 
        should_listen,
        thresh=0.3, 
        sample_rate=16000, 
        min_silence_ms=1000,
        min_speech_ms=500, 
        max_speech_ms=float('inf'),
        speech_pad_ms=30,
    ):
        self.should_listen = should_listen
        self.sample_rate = sample_rate
        self.min_silence_ms = min_silence_ms
        self.min_speech_ms = min_speech_ms
        self.max_speech_ms = max_speech_ms
        # Loading Silero VAD model from torch hub
        self.model, _ = torch.hub.load('snakers4/silero-vad', 'silero_vad', source='github')
        self.iterator = VADIterator(
            self.model,
            threshold=thresh,
            sampling_rate=sample_rate,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
        )

    def process(self, audio_chunk):
        # Convert audio_chunk to numpy array and float32
        audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
        audio_float32 = int2float(audio_int16)
        
        # Perform VAD (Voice Activity Detection)
        vad_output = self.iterator(torch.from_numpy(audio_float32))
        if vad_output is not None and len(vad_output) != 0:
            logger.debug("VAD: End of speech detected")
            # Concatenate and convert the output back to numpy array
            array = torch.cat(vad_output).cpu().numpy()
            duration_ms = len(array) / self.sample_rate * 1000
            
            # Check if the duration is within the speech length limits
            if duration_ms < self.min_speech_ms or duration_ms > self.max_speech_ms:
                logger.debug(f"Audio input of duration: {duration_ms/1000:.2f}s, skipping")
            else:
                # Stop listening once speech is detected
                self.should_listen.clear()
                logger.debug("Stop listening")
                yield array
