import numpy as np
import torch


def next_power_of_2(x: int) -> int:
    """
    Compute the next power of 2 greater than or equal to x.

    Parameters:
    ----------
    x : int
        Input integer value.

    Returns:
    -------
    int
        Next power of 2.
    """
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def int2float(sound: np.ndarray) -> np.ndarray:
    """
    Convert 16-bit PCM audio to float32 format and normalize.

    Parameters:
    ----------
    sound : np.ndarray
        Input 16-bit PCM audio array.

    Returns:
    -------
    np.ndarray
        Normalized float32 audio array.
    """
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1 / 32768  # Normalize the audio to [-1, 1] range
    return sound.squeeze()


class VADIterator:
    """
    Voice Activity Detector (VAD) iterator for speech segmentation based on a given model.
    
    Parameters:
    ----------
    model : torch.nn.Module
        Preloaded .jit or .onnx silero VAD model.

    threshold : float, optional (default=0.5)
        Probability threshold for classifying speech.

    sampling_rate : int, optional (default=16000)
        Sampling rate of the audio in Hz. Supports only 8000 or 16000.

    min_silence_duration_ms : int, optional (default=100)
        Minimum silence duration in milliseconds to end a speech segment.

    speech_pad_ms : int, optional (default=30)
        Padding (in milliseconds) added to each side of detected speech segments.
    """

    def __init__(self, model: torch.nn.Module, threshold: float = 0.5, sampling_rate: int = 16000, 
                 min_silence_duration_ms: int = 100, speech_pad_ms: int = 30):
        self.model = model
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.is_speaking = False
        self.buffer = []
        self.min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        self.speech_pad_samples = sampling_rate * speech_pad_ms / 1000
        self.reset_states()

        if sampling_rate not in [8000, 16000]:
            raise ValueError('VADIterator supports only 8000 or 16000 sampling rates')

    def reset_states(self) -> None:
        """
        Reset the internal states of the VAD model and buffer.
        """
        self.model.reset_states()
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an audio chunk and detect speech.

        Parameters:
        ----------
        x : torch.Tensor
            Input audio chunk to be processed.

        Returns:
        -------
        torch.Tensor or None
            Detected speech chunk if available, otherwise None.
        """
        if not torch.is_tensor(x):
            try:
                x = torch.Tensor(x)
            except:
                raise TypeError("Audio cannot be cast to tensor. Cast it manually.")

        window_size_samples = len(x[0]) if x.dim() == 2 else len(x)
        self.current_sample += window_size_samples

        speech_prob = self.model(x, self.sampling_rate).item()

        if speech_prob >= self.threshold and self.temp_end:
            self.temp_end = 0

        if speech_prob >= self.threshold and not self.triggered:
            self.triggered = True
            return None

        if speech_prob < self.threshold - 0.15 and self.triggered:
            if not self.temp_end:
                self.temp_end = self.current_sample
            if self.current_sample - self.temp_end < self.min_silence_samples:
                return None
            else:
                self.temp_end = 0
                self.triggered = False
                spoken_utterance = self.buffer
                self.buffer = []
                return torch.cat(spoken_utterance) if spoken_utterance else None

        if self.triggered:
            self.buffer.append(x)

        return None
