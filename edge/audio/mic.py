"""PC microphone capture with energy-based VAD + Whisper STT.

Runs in a background thread. Whenever it detects a completed utterance
(brief silence after speech), it transcribes and fires the callback.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

import numpy as np

log = logging.getLogger("mic")


class MicListener:
    def __init__(self,
                 on_text: Callable[[str], None],
                 sample_rate: int = 16000,
                 chunk_ms: int = 100,
                 silence_after_speech_s: float = 0.8,
                 max_utt_s: float = 8.0,
                 energy_thresh: int = 500,
                 stt=None) -> None:
        self.on_text = on_text
        self.sample_rate = sample_rate
        self.chunk_samples = int(sample_rate * chunk_ms / 1000)
        self.silence_after_speech_s = silence_after_speech_s
        self.max_utt_s = max_utt_s
        self.energy_thresh = energy_thresh
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._stt = stt  # lazy-loaded if None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="mic", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _ensure_stt(self):
        if self._stt is None:
            from audio.stt import WhisperSTT
            self._stt = WhisperSTT(model_size="tiny", language="zh")
        return self._stt

    def _run(self) -> None:
        try:
            import sounddevice as sd
        except ImportError:
            log.error("sounddevice not installed; mic disabled. pip install sounddevice")
            return

        stt = self._ensure_stt()
        buf: list[np.ndarray] = []
        speaking = False
        last_voice_t = 0.0
        utt_start = 0.0

        def cb(indata, frames, time_info, status):
            if status:
                log.debug("sd status: %s", status)

        log.info("mic listening at %d Hz, energy_thresh=%d", self.sample_rate, self.energy_thresh)
        try:
            with sd.InputStream(samplerate=self.sample_rate,
                                channels=1, dtype="int16",
                                blocksize=self.chunk_samples, callback=cb) as stream:
                while not self._stop.is_set():
                    block, _ = stream.read(self.chunk_samples)
                    pcm = block[:, 0].copy() if block.ndim == 2 else block.copy()
                    energy = int(np.abs(pcm).mean())
                    now = time.monotonic()

                    if energy > self.energy_thresh:
                        if not speaking:
                            speaking = True
                            utt_start = now
                            buf.clear()
                        last_voice_t = now
                        buf.append(pcm)
                    elif speaking:
                        buf.append(pcm)
                        silence = now - last_voice_t
                        too_long = now - utt_start > self.max_utt_s
                        if silence >= self.silence_after_speech_s or too_long:
                            audio = np.concatenate(buf) if buf else np.zeros(0, dtype=np.int16)
                            buf.clear()
                            speaking = False
                            if len(audio) > self.sample_rate * 0.3:
                                try:
                                    text = stt.transcribe_pcm(audio, self.sample_rate)
                                except Exception as e:
                                    log.warning("stt failed: %s", e)
                                    text = ""
                                text = (text or "").strip()
                                if text:
                                    log.info("heard: %s", text)
                                    try:
                                        self.on_text(text)
                                    except Exception:
                                        log.exception("on_text callback crashed")
        except Exception:
            log.exception("mic thread crashed")
