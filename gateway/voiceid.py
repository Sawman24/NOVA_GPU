"""Who's speaking: voice prints with SpeechBrain's ECAPA speaker model (runs on the GPU box's CPU:
a few milliseconds per sentence, and it avoids the GPU-generation issues of newer PyTorch builds).

Each person's voice print is a set of samples (one 192-number vector per sentence) in
data/gateway/voiceprints/<person>.npy. A new sentence is compared to each person's average.
Confident matches are added as new samples (up to MAX_SAMPLES), so recognition improves with use.

Install on the GPU box:  pip install speechbrain torch torchaudio --index-url ... (setup_gpu_box.sh does it)
Not installed -> status() says so and identify() returns nothing; everything else keeps working.
"""
from __future__ import annotations

import logging
import pathlib
import re
import threading
from typing import Any

import numpy as np

log = logging.getLogger("nova.voiceid")
MAX_SAMPLES = 40
MIN_SECONDS = 1.2        # shorter clips are too unreliable to identify
ENROLL_SECONDS = 2.0
LEARN_SECONDS = 2.5


class VoiceID:
    def __init__(self, cfg: dict[str, Any], data_dir: pathlib.Path):
        self.enabled = bool(cfg.get("enabled", True))
        self.sure = float(cfg.get("sure", 0.45))       # cosine score: at or above = recognised
        self.maybe = float(cfg.get("maybe", 0.30))     # between maybe and sure = "probably X"
        self.learn = float(cfg.get("learn", 0.55))     # at or above = add the sample to their print
        self.dir = data_dir / "voiceprints"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.model_dir = data_dir / "models" / "ecapa"
        self._model = None
        self._error = ""
        self._lock = threading.Lock()
        self._prints: dict[str, np.ndarray] = {}
        self._load_prints()

    # ------------------------------------------------------------------ prints
    @staticmethod
    def _safe(person: str) -> str:
        p = re.sub(r"[^a-z0-9_-]", "", person.lower())
        if not p:
            raise ValueError("bad person id")
        return p

    def _load_prints(self) -> None:
        self._prints = {}
        for f in self.dir.glob("*.npy"):
            try:
                a = np.load(f)
                if a.ndim == 2 and len(a):
                    self._prints[f.stem] = a.astype(np.float32)
            except (OSError, ValueError) as e:
                log.warning("bad voice print %s: %s", f.name, e)

    def _save(self, person: str) -> None:
        np.save(self.dir / f"{person}.npy", self._prints[person])

    def people(self) -> dict[str, int]:
        return {p: int(len(a)) for p, a in sorted(self._prints.items())}

    def forget(self, person: str) -> bool:
        person = self._safe(person)
        with self._lock:
            had = self._prints.pop(person, None) is not None
            (self.dir / f"{person}.npy").unlink(missing_ok=True)
        return had

    # ------------------------------------------------------------------ model
    def installed(self) -> bool:
        try:
            import speechbrain  # noqa: F401
            import torch  # noqa: F401
            return True
        except Exception:  # noqa: BLE001
            return False

    def status(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "installed": self.installed(), "loaded": self._model is not None,
                "error": self._error, "people": self.people(),
                "thresholds": {"sure": self.sure, "maybe": self.maybe}}

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            try:
                from speechbrain.inference.speaker import EncoderClassifier
            except ImportError:  # speechbrain < 1.0
                from speechbrain.pretrained import EncoderClassifier
            self._model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(self.model_dir), run_opts={"device": "cpu"})
            self._error = ""
            log.info("Voice recognition model loaded")
        except Exception as e:  # noqa: BLE001
            self._error = f"{type(e).__name__}: {e}"
            log.warning("Voice recognition unavailable: %s", self._error)
            raise
        return self._model

    def embed(self, pcm: np.ndarray) -> np.ndarray:
        import torch
        model = self._load()
        with torch.no_grad():
            e = model.encode_batch(torch.from_numpy(np.ascontiguousarray(pcm, dtype=np.float32)).unsqueeze(0))
        v = e.squeeze().cpu().numpy().astype(np.float32)
        return v / max(float(np.linalg.norm(v)), 1e-9)

    @staticmethod
    def _speech_only(pcm: np.ndarray) -> np.ndarray:
        """Drop near-silent 30 ms frames so pauses don't dilute the print."""
        n = 480
        if len(pcm) < n * 4:
            return pcm
        frames = pcm[: len(pcm) // n * n].reshape(-1, n)
        rms = np.sqrt((frames ** 2).mean(axis=1))
        keep = rms > max(0.006, float(np.percentile(rms, 30)) * 1.2)
        return frames[keep].reshape(-1) if keep.sum() >= 10 else pcm

    # ------------------------------------------------------------------ use
    def enroll(self, person: str, pcm: np.ndarray) -> dict[str, Any]:
        person = self._safe(person)
        speech = self._speech_only(pcm)
        if len(speech) < ENROLL_SECONDS * 16000:
            raise ValueError(f"Need at least {ENROLL_SECONDS:.0f} seconds of speech; got {len(speech) / 16000:.1f}.")
        with self._lock:
            v = self.embed(speech)
            cur = self._prints.get(person)
            self._prints[person] = (np.vstack([cur, v]) if cur is not None else v[None, :])[-MAX_SAMPLES:]
            self._save(person)
            n = len(self._prints[person])
        return {"person": person, "samples": n}

    def identify(self, pcm: np.ndarray) -> dict[str, Any] | None:
        """{person, score, confidence: sure|maybe|unknown, runner_up, seconds} or None if not possible."""
        if not self.enabled or not self._prints or not self.installed():
            return None
        speech = self._speech_only(pcm)
        secs = len(speech) / 16000
        if secs < MIN_SECONDS:
            return {"person": "", "score": 0.0, "confidence": "too_short", "seconds": round(secs, 1)}
        with self._lock:
            try:
                v = self.embed(speech)
            except Exception:  # noqa: BLE001
                return None
            scores = {}
            for p, samples in self._prints.items():
                c = samples.mean(axis=0)
                c /= max(float(np.linalg.norm(c)), 1e-9)
                # mean-print score, nudged by the best single sample (helps with varied recordings)
                scores[p] = 0.7 * float(c @ v) + 0.3 * float((samples @ v).max())
            ranked = sorted(scores.items(), key=lambda kv: -kv[1])
            best, score = ranked[0]
            second = ranked[1] if len(ranked) > 1 else ("", 0.0)
            # A near-tie between two people is not "sure".
            margin_ok = score - second[1] >= 0.08
            conf = "sure" if score >= self.sure and margin_ok else "maybe" if score >= self.maybe else "unknown"
            if conf == "sure" and score >= self.learn and secs >= LEARN_SECONDS:
                arr = np.vstack([self._prints[best], v])[-MAX_SAMPLES:]
                self._prints[best] = arr
                self._save(best)
        return {"person": best if conf != "unknown" else "", "score": round(score, 3), "confidence": conf,
                "runner_up": {"person": second[0], "score": round(second[1], 3)} if second[0] else None,
                "seconds": round(secs, 1)}
