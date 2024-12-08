import librosa
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

class ASRModel:
    def __init__(self, model_name='facebook/wav2vec2-base-960h'):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)

    def transcribe(self, audio_path: str) -> str:
        try:
            audio, rate = librosa.load(audio_path, sr=16000)
            input_values = self.processor(audio, sampling_rate=rate, return_tensors="pt").input_values
            logits = self.model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
            return transcription
        except Exception as e:
            return f"Error in transcription: {str(e)}"
