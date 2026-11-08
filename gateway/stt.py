"""Speech to text with faster-whisper (runs on the GPU box).

device "auto" tries the GPU first and falls back to the CPU if CUDA libraries are
missing (common on Windows). The model loads on first use.
"""
from __future__ import annotations

import io
import logging
import threading
import time
from typing import Any

log = logging.getLogger("nova.stt")


class SpeechToText:
    def __init__(self, cfg: dict[str, Any]):
        self.model_name = str(cfg.get("model", "small.en"))
        self.device = str(cfg.get("device", "auto"))
        self.compute_type = str(cfg.get("compute_type", "auto"))
        self.language = cfg.get("language", "en") or None
        self.beam_size = int(cfg.get("beam_size", 5))
        self.enabled = bool(cfg.get("enabled", True))
        self._model = None
        self._active_device = ""
        self._error = ""
        self._lock = threading.Lock()

    def status(self) -> dict[str, Any]:
        try:
            import faster_whisper  # noqa: F401
            installed = True
        except ImportError:
            installed = False
        return {
            "enabled": self.enabled,
            "installed": installed,
            "model": self.model_name,
            "loaded": self._model is not None,
            "device": self._active_device,
            "error": self._error,
        }

    @staticmethod
    def _windows_cuda_dlls() -> None:
        """Let CTranslate2 find cuBLAS/cuDNN installed as pip packages (nvidia-*-cu12) on Windows."""
        import os
        import site
        import sys

        if os.name != "nt":
            return
        roots = site.getsitepackages() + [site.getusersitepackages(), os.path.join(sys.prefix, "Lib", "site-packages")]
        for root in roots:
            nv = os.path.join(root, "nvidia")
            if not os.path.isdir(nv):
                continue
            for pkg in os.listdir(nv):
                b = os.path.join(nv, pkg, "bin")
                if os.path.isdir(b):
                    try:
                        os.add_dll_directory(b)
                    except OSError:
                        pass
                    os.environ["PATH"] = b + os.pathsep + os.environ.get("PATH", "")

    def _load(self):
        self._windows_cuda_dlls()
        from faster_whisper import WhisperModel
        import numpy as np

        attempts = []
        if self.device in ("auto", "cuda"):
            attempts.append(("cuda", "float16" if self.compute_type == "auto" else self.compute_type))
        if self.device in ("auto", "cpu"):
            attempts.append(("cpu", "int8" if self.compute_type == "auto" else self.compute_type))
        last = None
        for device, ctype in attempts:
            try:
                t = time.time()
                model = WhisperModel(self.model_name, device=device, compute_type=ctype)
                # A real inference catches missing CUDA DLLs that only fail at run time.
                list(model.transcribe(np.zeros(16000, dtype=np.float32), beam_size=1)[0])
                self._model, self._active_device, self._error = model, f"{device}/{ctype}", ""
                log.info("Whisper %s loaded on %s/%s in %.1fs", self.model_name, device, ctype, time.time() - t)
                return
            except Exception as e:  # noqa: BLE001
                last = e
                log.warning("Whisper on %s failed (%s); trying next option", device, e)
        self._error = str(last)
        raise RuntimeError(f"Could not load Whisper: {last}")

    @staticmethod
    def decode(audio: bytes):
        """Bytes -> 16 kHz mono float32. WAV is read directly; anything else (webm, ogg, mp3) via PyAV.
        Done here rather than in faster-whisper, whose decoder breaks with newer PyAV releases."""
        import wave

        import numpy as np

        if audio[:4] == b"RIFF":
            with wave.open(io.BytesIO(audio), "rb") as w:
                rate, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
                raw = w.readframes(w.getnframes())
            dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
            pcm = np.frombuffer(raw, dtype=dtype).astype(np.float32)
            if width == 1:
                pcm = (pcm - 128) / 128
            else:
                pcm /= float(2 ** (8 * width - 1))
            if ch > 1:
                pcm = pcm.reshape(-1, ch).mean(axis=1)
        else:
            import av

            chunks = []
            with av.open(io.BytesIO(audio), mode="r") as container:
                resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
                for frame in container.decode(audio=0):
                    for f in resampler.resample(frame):
                        chunks.append(f.to_ndarray().reshape(-1))
                for f in resampler.resample(None):
                    chunks.append(f.to_ndarray().reshape(-1))
            pcm = (np.concatenate(chunks) if chunks else np.zeros(0, np.int16)).astype(np.float32) / 32768.0
            rate = 16000
        if rate != 16000 and len(pcm):
            n = int(len(pcm) * 16000 / rate)
            pcm = np.interp(np.linspace(0, len(pcm) - 1, n), np.arange(len(pcm)), pcm).astype(np.float32)
        return pcm.astype(np.float32)

    def transcribe(self, audio) -> dict[str, Any]:
        """audio: bytes in any format PyAV can decode (wav, webm, ogg, mp3...), or 16 kHz float32 samples."""
        if not self.enabled:
            raise RuntimeError("Speech to text is disabled in nova.yaml")
        with self._lock:
            if self._model is None:
                self._load()
            t = time.time()
            pcm = self.decode(audio) if isinstance(audio, (bytes, bytearray)) else audio
            if len(pcm) < 1600:  # under 0.1 s
                return {"text": "", "language": self.language, "duration": 0.0, "seconds": 0.0}
            segments, info = self._model.transcribe(
                pcm,
                language=self.language,
                beam_size=self.beam_size,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            text = " ".join(s.text.strip() for s in segments).strip()
        return {
            "text": text,
            "language": getattr(info, "language", self.language),
            "duration": round(float(getattr(info, "duration", 0.0)), 2),
            "seconds": round(time.time() - t, 2),
        }
