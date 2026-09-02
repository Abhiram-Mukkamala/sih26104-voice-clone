"""
acoustic_model.py
==================
Track A wrapper: loads an INT8-quantized ONNX acoustic model (AASIST /
RawNet3 / WavLM-based spoof classifier) and exposes a single
`predict_vocoder_probability(lfcc)` call.

This repo ships without a trained checkpoint (no SIH team has one on day
one) — if `model_path` is missing, the class transparently falls back to
a deterministic, explainable DSP heuristic built from the same LFCC
tensor, so the rest of the pipeline (fusion, server, WebSocket contract)
is fully runnable and demoable end-to-end before the model is trained.
Swap in a real .onnx file and nothing else in the codebase changes.
"""

from __future__ import annotations

import os
import numpy as np

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _ORT_AVAILABLE = False


class AcousticModel:
    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        self.session = None
        self.input_name = None

        if model_path and _ORT_AVAILABLE and os.path.exists(model_path):
            # INT8 quantized model -> CPUExecutionProvider is sufficient for
            # sub-250ms budget at this input size; add 'CUDAExecutionProvider'
            # first in the list if a GPU is available on the deployment box.
            self.session = ort.InferenceSession(
                model_path, providers=["CPUExecutionProvider"]
            )
            self.input_name = self.session.get_inputs()[0].name

    @property
    def using_fallback(self) -> bool:
        return self.session is None

    def predict_vocoder_probability(self, lfcc: np.ndarray) -> float:
        """
        lfcc: (n_frames, n_lfcc) float32 array from dsp_pipeline.compute_lfcc.
        Returns P(vocoder/synthetic) in [0, 1].
        """
        if self.session is not None:
            batch = lfcc[np.newaxis, :, :].astype(np.float32)  # (1, T, F)
            outputs = self.session.run(None, {self.input_name: batch})
            logits = outputs[0].squeeze()
            # Assumes a 2-logit [bonafide, spoof] head; adjust to your
            # checkpoint's actual output contract at integration time.
            probs = _softmax(logits)
            return float(probs[-1])

        return self._fallback_heuristic(lfcc)

    @staticmethod
    def _fallback_heuristic(lfcc: np.ndarray) -> float:
        """
        Explainable stand-in used only when no trained checkpoint is
        loaded. Vocoder upsampling artifacts show up as excess energy and
        low frame-to-frame variance in the *higher* cepstral coefficients
        (fine spectral detail) relative to the low-order coefficients
        (coarse envelope/timbre). We turn that ratio into a bounded score.
        This is intentionally conservative — it is a placeholder for a
        trained AASIST/RawNet3 head, not a claimed detector.
        """
        if lfcc.size == 0:
            return 0.5

        low_order = lfcc[:, 1:6]     # coarse spectral envelope
        high_order = lfcc[:, 12:20]  # fine spectral detail

        high_energy = np.mean(np.abs(high_order))
        low_energy = np.mean(np.abs(low_order)) + 1e-6
        energy_ratio = high_energy / low_energy

        high_frame_var = np.mean(np.var(high_order, axis=0))
        # Unnaturally low variance in fine detail across time -> suspicious.
        smoothness_penalty = 1.0 / (1.0 + high_frame_var)

        raw_score = 0.5 * np.tanh(energy_ratio - 0.6) + 0.5 * smoothness_penalty
        return float(np.clip((raw_score + 1) / 2, 0.0, 1.0))


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)
