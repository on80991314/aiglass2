"""Speech-to-text wrapper using Groq API (replaces local faster-whisper to prevent crashes)."""

from __future__ import annotations
import numpy as np
import io
import wave
from groq import Groq

class WhisperSTT:
    def __init__(self, model_size: str = "tiny", device: str = "cpu",
                 compute_type: str = "int8", language: str = "zh") -> None:
        self.language = language
        # 沿用你現有的 Groq API Key
        self.client = Groq(api_key="GROQ_API_KEY")

    def transcribe_pcm(self, pcm16_mono: np.ndarray, sample_rate: int = 16000) -> str:
        """將 PCM 音訊轉為 WAV 格式，並發送給 Groq 雲端 Whisper 辨識"""
        # 將 numpy 陣列轉換為記憶體中的 WAV 檔案，不佔用硬碟
        wav_io = io.BytesIO()
        with wave.open(wav_io, 'wb') as wf:
            wf.setnchannels(1)  # 單聲道
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16_mono.tobytes())
        
        # 準備發送給 Groq
        wav_io.seek(0)
        wav_io.name = "audio.wav"
        
        try:
            # 使用 Groq 的超高速 Whisper API
            transcription = self.client.audio.transcriptions.create(
                file=("audio.wav", wav_io.read()),
                model="whisper-large-v3-turbo", # 雲端超大模型，速度快又準
                language=self.language,
            )
            return transcription.text.strip()
        except Exception as e:
            print(f"[Groq STT Error] 語音辨識失敗: {e}")
            return ""