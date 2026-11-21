import os
import json
import math
import time
import threading
import numpy as np
from flask import Flask, render_template, request, jsonify, Response, stream_with_context

app = Flask(__name__)
STATE_LOCK = threading.Lock()


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.maximum(1e-12, np.sum(exps, axis=-1, keepdims=True))


# ==============================================================================
# 1. REAL SPIKING DATASET GENERATORS (Self-Contained Scientific Tasks)
# ==============================================================================
class CueAccumulationDatasetGenerator:
    """
    Task 1: Sequential Evidence Accumulation (Bellec et al., 2020)
    - 40 input channels: [0..9] Left cue, [10..19] Right cue, [20..29] Recall prompt, [30..39] Background noise.
    - 5 sequential sensory cues followed by a silent memory delay (100-800 ms) and a recall decision window.
    """
    def __init__(self, n_ch_per_group=10, num_cues=5, dt_ms=5.0,
                 cue_on_ms=25.0, cue_gap_ms=15.0, recall_ms=50.0,
                 cue_rate_hz=80.0, bg_rate_hz=10.0):
        self.n_ch = n_ch_per_group
        self.input_dim = 4 * n_ch_per_group
        self.num_cues = num_cues
        self.dt_ms = dt_ms
        self.dt = dt_ms * 1e-3
        self.num_classes = 2
        self.class_names = ["Left Majority (Class 0)", "Right Majority (Class 1)"]

        self.cue_on_steps = int(cue_on_ms / dt_ms)
        self.cue_slot_steps = int((cue_on_ms + cue_gap_ms) / dt_ms)
        self.recall_steps = int(recall_ms / dt_ms)

        self.p_cue = min(0.92, cue_rate_hz * self.dt)
        self.p_bg = min(0.40, bg_rate_hz * self.dt)

    def sample_batch(self, batch_size: int, delay_ms: float, seed: int = None):
        rng = np.random.default_rng(seed)
        delay_steps = int(round(delay_ms / self.dt_ms))
        cues_total_steps = self.num_cues * self.cue_slot_steps
        recall_start = cues_total_steps + delay_steps
        total_steps = recall_start + self.recall_steps

        probs = np.full((batch_size, total_steps, self.input_dim), self.p_bg, dtype=np.float32)
        right_counts = np.zeros(batch_size, dtype=np.int64)

        majority_side = rng.integers(0, 2, size=batch_size)
        p_right = np.where(majority_side == 1, 0.82, 0.18)

        for k in range(self.num_cues):
            t0 = k * self.cue_slot_steps
            t1 = t0 + self.cue_on_steps
            is_right = (rng.random(batch_size) < p_right).astype(np.int64)
            right_counts += is_right
            for b in range(batch_size):
                if is_right[b] == 1:
                    probs[b, t0:t1, self.n_ch : 2 * self.n_ch] += self.p_cue
                else:
                    probs[b, t0:t1, 0 : self.n_ch] += self.p_cue

        labels = (right_counts > (self.num_cues // 2)).astype(np.int64)
        probs[:, recall_start:total_steps, 2 * self.n_ch : 3 * self.n_ch] += self.p_cue
        spikes = (rng.random(probs.shape) < np.clip(probs, 0.0, 0.95)).astype(np.float32)
        return spikes, labels, recall_start, total_steps, cues_total_steps


class DelayedCueAssociationDatasetGenerator:
    """
    Task 2: Delayed Cue-Target Association & Hippocampal Comparator
    - Input channels: 3 Cue Classes (8 channels each = 24) + Go/Recall channels (8 channels) + Noise (8 channels) = 40.
    - Cue Window [0, 50 ms) -> Silent Delay (delay_ms) -> Target / Recall Window [recall_start, total_steps).
    """
    def __init__(self, num_classes=3, n_ch_per_group=8, dt_ms=5.0,
                 cue_dur_ms=50.0, recall_ms=60.0,
                 cue_rate_hz=90.0, bg_rate_hz=8.0):
        self.num_classes = num_classes
        self.n_ch = n_ch_per_group
        self.input_dim = (num_classes + 2) * n_ch_per_group
        self.dt_ms = dt_ms
        self.dt = dt_ms * 1e-3
        self.class_names = ["Cue A -> Target A (Class 0)", "Cue B -> Target B (Class 1)", "Cue C -> Target C (Class 2)"]

        self.cue_steps = int(cue_dur_ms / dt_ms)
        self.recall_steps = int(recall_ms / dt_ms)
        self.p_cue = min(0.92, cue_rate_hz * self.dt)
        self.p_bg = min(0.40, bg_rate_hz * self.dt)

    def sample_batch(self, batch_size: int, delay_ms: float, seed: int = None):
        rng = np.random.default_rng(seed)
        delay_steps = int(round(delay_ms / self.dt_ms))
        cues_total_steps = self.cue_steps
        recall_start = cues_total_steps + delay_steps
        total_steps = recall_start + self.recall_steps

        labels = rng.integers(0, self.num_classes, size=batch_size, dtype=np.int64)
        probs = np.full((batch_size, total_steps, self.input_dim), self.p_bg, dtype=np.float32)

        for b in range(batch_size):
            c = int(labels[b])
            probs[b, :self.cue_steps, c * self.n_ch : (c + 1) * self.n_ch] += self.p_cue

        go_ch_start = self.num_classes * self.n_ch
        probs[:, recall_start:total_steps, go_ch_start : go_ch_start + self.n_ch] += self.p_cue

        spikes = (rng.random(probs.shape) < np.clip(probs, 0.0, 0.95)).astype(np.float32)
        return spikes, labels, recall_start, total_steps, cues_total_steps


# ==============================================================================
# 2. UNIFIED 6-MODEL RECURRENT SNN CATALOG (Zero Internal Notebook Jargon)
# ==============================================================================
MODEL_CATALOG = {
    "lif_eprop": {
        "id": "lif_eprop",
        "name": "M1: Standard LIF (Online e-prop)",
        "role_badge": "Fast Trace Baseline",
        "cell_type": "lif",
        "rule": "eprop",
        "color": "#64748b",
        "description": "Standard Leaky Integrate-and-Fire network trained online with fast voltage eligibility traces (tau_m = 20 ms). Memory fades over long silent delays.",
    },
    "lif_theta_eprop": {
        "id": "lif_theta_eprop",
        "name": "M2: LIF + Theta-Gamma Oscillation (Online e-prop)",
        "role_badge": "8Hz / 40Hz Phase Gating",
        "cell_type": "lif_theta",
        "rule": "eprop",
        "color": "#3b82f6",
        "description": "Recurrent LIF modulated by coherent 8 Hz Theta + 40 Hz Gamma oscillations, organizing spikes into phase-locked rhythmic windows.",
    },
    "alif_eprop": {
        "id": "alif_eprop",
        "name": "M3: Adaptive LIF / LSNN (Online e-prop)",
        "role_badge": "Slow Threshold Adaptation",
        "cell_type": "alif",
        "rule": "eprop",
        "color": "#10b981",
        "description": "Adaptive LIF (Bellec et al., 2020) with slow firing-threshold adaptation (tau_a = 500 ms) that carries a slow eligibility trace across silent delays.",
    },
    "alif_theta_eprop": {
        "id": "alif_theta_eprop",
        "name": "M4: ALIF + Theta-Gamma Hybrid (Online e-prop)",
        "role_badge": "Cellular + Network Synergy",
        "cell_type": "alif_theta",
        "rule": "eprop",
        "color": "#f59e0b",
        "description": "Combines slow cellular threshold adaptation (tau_a) with network-level 8 Hz / 40 Hz Theta-Gamma oscillatory phase gating.",
    },
    "alif_bptt": {
        "id": "alif_bptt",
        "name": "M5: ALIF / LIF (Offline Surrogate BPTT Ceiling)",
        "role_badge": "Offline BPTT Upper Bound",
        "cell_type": "alif_theta",
        "rule": "bptt",
        "color": "#a855f7",
        "description": "Unrolled Backpropagation Through Time storing the full O(N*T) state history across the entire trial (non-biological upper-bound baseline).",
    },
    "lif_random_phase": {
        "id": "lif_random_phase",
        "name": "M6: Control — Desynchronized Random Theta Phase",
        "role_badge": "Unaligned Phase Control",
        "cell_type": "lif_random_phase",
        "rule": "eprop",
        "color": "#ef4444",
        "description": "Receives the same 8 Hz Theta-Gamma amplitude, but with randomized phase jitter — tests whether coherent phase alignment is causally required.",
    },
}


class NumpyAdamOptimizer:
    """Dependency-free Adam optimizer with global L2 gradient norm clipping."""
    def __init__(self, params: dict, lr=0.004, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads: dict, max_norm=2.0):
        self.t += 1
        total_norm_sq = sum(float(np.sum(g * g)) for g in grads.values())
        total_norm = math.sqrt(max(1e-12, total_norm_sq))
        clip_coef = min(1.0, max_norm / total_norm)

        for k, p in self.params.items():
            g = grads[k] * clip_coef
            self.m[k] = self.beta1 * self.m[k] + (1.0 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1.0 - self.beta2) * (g * g)
            m_hat = self.m[k] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1.0 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class VisualizerRecurrentSNN:
    """
    Vectorized Recurrent Spiking Neural Network supporting all 6 architectures:
      - Exact Bellec et al. (2020) online forward-in-time e-prop (Eqs. 13-28)
      - Exact unrolled Backpropagation Through Time (BPTT) with dampened triangle surrogate gradients
      - Millisecond-by-millisecond telemetry recorder for the Interactive Model Visualizer
    """
    def __init__(self, input_dim=40, hidden_dim=64, num_classes=2, cell_type="alif_theta",
                 dt_ms=5.0, tau_mem_ms=20.0, tau_adapt_ms=500.0, tau_out_ms=20.0,
                 v_thresh=0.6, beta_adapt=1.6, alif_fraction=0.5, gamma_pd=0.3,
                 f_theta=8.0, f_gamma=40.0, a_theta=0.15, m_gamma=0.25, seed=42):
        rng = np.random.default_rng(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.cell_type = cell_type
        self.dt_ms = dt_ms
        self.dt = dt_ms * 1e-3

        self.alpha = float(math.exp(-dt_ms / max(1.0, tau_mem_ms)))
        self.rho = float(math.exp(-dt_ms / max(10.0, tau_adapt_ms)))
        self.kappa = float(math.exp(-dt_ms / max(1.0, tau_out_ms)))
        self.v_thresh = float(v_thresh)
        self.gamma_pd = float(gamma_pd)

        self.f_theta = float(f_theta)
        self.f_gamma = float(f_gamma)
        self.a_theta = float(a_theta)
        self.m_gamma = float(m_gamma)

        lim_in = math.sqrt(6.0 / (input_dim + hidden_dim)) * 0.85
        self.w_in = rng.uniform(-lim_in, lim_in, (hidden_dim, input_dim)).astype(np.float32)

        raw_rec = rng.normal(0.0, 1.0, (hidden_dim, hidden_dim)).astype(np.float32)
        u, _, vt = np.linalg.svd(raw_rec, full_matrices=False)
        self.w_rec = (u @ vt * 0.78).astype(np.float32)
        np.fill_diagonal(self.w_rec, 0.0)

        lim_out = math.sqrt(6.0 / (hidden_dim + num_classes)) * 0.80
        self.w_out = rng.uniform(-lim_out, lim_out, (num_classes, hidden_dim)).astype(np.float32)
        self.b_out = np.zeros(num_classes, dtype=np.float32)

        self.beta_vec = np.zeros(hidden_dim, dtype=np.float32)
        if "alif" in cell_type:
            n_a = int(alif_fraction * hidden_dim)
            self.beta_vec[:n_a] = float(beta_adapt)

        self.phase_pref = np.linspace(0.0, 2.0 * math.pi, hidden_dim, dtype=np.float32)

    def parameters(self) -> dict:
        return {
            "w_in": self.w_in,
            "w_rec": self.w_rec,
            "w_out": self.w_out,
            "b_out": self.b_out,
        }

    def get_oscillation(self, batch_size: int, total_steps: int, seed: int = 0) -> np.ndarray:
        if "theta" not in self.cell_type and "random_phase" not in self.cell_type:
            return np.zeros((batch_size, total_steps, self.hidden_dim), dtype=np.float32)

        t = (np.arange(total_steps, dtype=np.float32) * self.dt).reshape(1, total_steps, 1)
        pref = self.phase_pref.reshape(1, 1, self.hidden_dim)

        if "random_phase" in self.cell_type:
            rng = np.random.default_rng(seed + 999)
            phi0 = rng.uniform(0.0, 2.0 * math.pi, size=(batch_size, 1, 1)).astype(np.float32)
            jitter = rng.normal(0.0, 1.15, size=(batch_size, total_steps, 1)).astype(np.float32)
        else:
            phi0 = 0.0
            jitter = 0.0

        theta = np.sin(2.0 * math.pi * self.f_theta * t + phi0 + 0.3 * pref + jitter)
        gamma = 1.0 + self.m_gamma * np.sin(2.0 * math.pi * self.f_gamma * t)
        osc = (self.a_theta * theta * gamma).astype(np.float32)
        if osc.shape[0] == 1 and batch_size > 1:
            osc = np.broadcast_to(osc, (batch_size, total_steps, self.hidden_dim)).copy()
        return osc

    def forward_eprop(self, x: np.ndarray, y: np.ndarray, recall_start: int, compute_grads: bool = True, seed: int = 0):
        B, T, I = x.shape
        H = self.hidden_dim
        C = self.num_classes
        i_osc = self.get_oscillation(B, T, seed=seed)

        y_onehot = np.zeros((B, C), dtype=np.float32)
        y_onehot[np.arange(B), y] = 1.0

        v = np.zeros((B, H), dtype=np.float32)
        a = np.zeros((B, H), dtype=np.float32)
        z = np.zeros((B, H), dtype=np.float32)
        y_out = np.zeros((B, C), dtype=np.float32)

        eps_v_in = np.zeros((B, I), dtype=np.float32)
        eps_v_rec = np.zeros((B, H), dtype=np.float32)

        has_alif = "alif" in self.cell_type
        if has_alif:
            eps_a_in = np.zeros((B, H, I), dtype=np.float32)
            eps_a_rec = np.zeros((B, H, H), dtype=np.float32)

        ebar_in = np.zeros((B, H, I), dtype=np.float32)
        ebar_rec = np.zeros((B, H, H), dtype=np.float32)
        z_filt_out = np.zeros((B, H), dtype=np.float32)

        grad_w_in = np.zeros_like(self.w_in)
        grad_w_rec = np.zeros_like(self.w_rec)
        grad_w_out = np.zeros_like(self.w_out)
        grad_b_out = np.zeros_like(self.b_out)

        loss_accum = 0.0
        recall_logits_sum = np.zeros((B, C), dtype=np.float32)
        recall_steps = max(1, T - recall_start)
        beta_row = self.beta_vec.reshape(1, H)
        beta_col = self.beta_vec.reshape(1, H, 1)

        for t in range(T):
            z_prev = z
            x_t = x[:, t, :]

            eps_v_in = self.alpha * eps_v_in + x_t
            eps_v_rec = self.alpha * eps_v_rec + z_prev

            thr_t = self.v_thresh + beta_row * a
            i_syn = x_t @ self.w_in.T + z_prev @ self.w_rec.T + i_osc[:, t, :]
            v = self.alpha * v + i_syn - z_prev * self.v_thresh

            v_scaled = (v - thr_t) / self.v_thresh
            z = (v_scaled >= 0.0).astype(np.float32)

            if compute_grads:
                psi = (self.gamma_pd / self.v_thresh) * np.maximum(0.0, 1.0 - np.abs(v_scaled))
                psi_col = psi[:, :, None]
                ev_in_3d = eps_v_in[:, None, :]
                ev_rec_3d = eps_v_rec[:, None, :]

                if has_alif:
                    e_in = psi_col * (ev_in_3d - beta_col * eps_a_in)
                    e_rec = psi_col * (ev_rec_3d - beta_col * eps_a_rec)
                    eps_a_in = psi_col * ev_in_3d + (self.rho - psi_col * beta_col) * eps_a_in
                    eps_a_rec = psi_col * ev_rec_3d + (self.rho - psi_col * beta_col) * eps_a_rec
                else:
                    e_in = psi_col * ev_in_3d
                    e_rec = psi_col * ev_rec_3d

                ebar_in = self.kappa * ebar_in + e_in
                ebar_rec = self.kappa * ebar_rec + e_rec
                z_filt_out = self.kappa * z_filt_out + z

            if has_alif:
                a = self.rho * a + z

            y_out = self.kappa * y_out + (z @ self.w_out.T + self.b_out)
            if t >= recall_start:
                pi_t = softmax_np(y_out)
                ce = -np.mean(np.sum(y_onehot * np.log(np.maximum(1e-12, pi_t)), axis=-1))
                loss_accum += float(ce)
                recall_logits_sum += y_out

                if compute_grads:
                    delta_out = (pi_t - y_onehot) / float(B * recall_steps)
                    L_t = delta_out @ self.w_out

                    grad_w_out += delta_out.T @ z_filt_out
                    grad_b_out += np.sum(delta_out, axis=0)
                    grad_w_in += np.einsum("bh,bhi->hi", L_t, ebar_in)
                    grad_w_rec += np.einsum("bh,bhj->hj", L_t, ebar_rec)

        np.fill_diagonal(grad_w_rec, 0.0)
        preds = np.argmax(recall_logits_sum, axis=-1)
        acc = float(np.mean(preds == y)) * 100.0
        grads = {
            "w_in": grad_w_in,
            "w_rec": grad_w_rec,
            "w_out": grad_w_out,
            "b_out": grad_b_out,
        }
        return loss_accum / float(recall_steps), acc, grads

    def forward_bptt(self, x: np.ndarray, y: np.ndarray, recall_start: int, seed: int = 0):
        B, T, I = x.shape
        H = self.hidden_dim
        C = self.num_classes
        i_osc = self.get_oscillation(B, T, seed=seed)

        y_onehot = np.zeros((B, C), dtype=np.float32)
        y_onehot[np.arange(B), y] = 1.0

        v_hist = np.zeros((T, B, H), dtype=np.float32)
        z_hist = np.zeros((T + 1, B, H), dtype=np.float32)
        psi_hist = np.zeros((T, B, H), dtype=np.float32)
        y_out_hist = np.zeros((T, B, C), dtype=np.float32)

        v = np.zeros((B, H), dtype=np.float32)
        a = np.zeros((B, H), dtype=np.float32)
        z = np.zeros((B, H), dtype=np.float32)
        y_out = np.zeros((B, C), dtype=np.float32)

        has_alif = "alif" in self.cell_type
        beta_row = self.beta_vec.reshape(1, H)
        loss_accum = 0.0
        recall_logits_sum = np.zeros((B, C), dtype=np.float32)
        recall_steps = max(1, T - recall_start)

        for t in range(T):
            z_hist[t] = z
            thr_t = self.v_thresh + beta_row * a
            i_syn = x[:, t, :] @ self.w_in.T + z @ self.w_rec.T + i_osc[:, t, :]
            v = self.alpha * v + i_syn - z * self.v_thresh
            v_hist[t] = v

            v_scaled = (v - thr_t) / self.v_thresh
            z = (v_scaled >= 0.0).astype(np.float32)
            psi_hist[t] = (self.gamma_pd / self.v_thresh) * np.maximum(0.0, 1.0 - np.abs(v_scaled))

            if has_alif:
                a = self.rho * a + z

            y_out = self.kappa * y_out + (z @ self.w_out.T + self.b_out)
            y_out_hist[t] = y_out
            if t >= recall_start:
                pi_t = softmax_np(y_out)
                ce = -np.mean(np.sum(y_onehot * np.log(np.maximum(1e-12, pi_t)), axis=-1))
                loss_accum += float(ce)
                recall_logits_sum += y_out

        z_hist[T] = z

        grad_w_in = np.zeros_like(self.w_in)
        grad_w_rec = np.zeros_like(self.w_rec)
        grad_w_out = np.zeros_like(self.w_out)
        grad_b_out = np.zeros_like(self.b_out)

        dy_out_next = np.zeros((B, C), dtype=np.float32)
        dv_next = np.zeros((B, H), dtype=np.float32)
        da_next = np.zeros((B, H), dtype=np.float32)

        for t in range(T - 1, -1, -1):
            dy_out = self.kappa * dy_out_next
            if t >= recall_start:
                pi_t = softmax_np(y_out_hist[t])
                dy_out = dy_out + (pi_t - y_onehot) / float(B * recall_steps)
            dy_out_next = dy_out

            grad_w_out += dy_out.T @ z_hist[t + 1]
            grad_b_out += np.sum(dy_out, axis=0)

            dz = dy_out @ self.w_out + dv_next @ self.w_rec
            if has_alif:
                dz = dz + da_next

            dv = dz * psi_hist[t] + self.alpha * dv_next
            if has_alif:
                da = self.rho * da_next - dz * psi_hist[t] * beta_row
                da_next = da
            dv_next = dv

            grad_w_in += dv.T @ x[:, t, :]
            grad_w_rec += dv.T @ z_hist[t]

        np.fill_diagonal(grad_w_rec, 0.0)
        preds = np.argmax(recall_logits_sum, axis=-1)
        acc = float(np.mean(preds == y)) * 100.0
        grads = {
            "w_in": grad_w_in,
            "w_rec": grad_w_rec,
            "w_out": grad_w_out,
            "b_out": grad_b_out,
        }
        return loss_accum / float(recall_steps), acc, grads

    def record_single_trial_telemetry(self, x_single: np.ndarray, y_single: int,
                                      recall_start: int, cues_end: int, omit_target: bool = False):
        _, T, I = x_single.shape
        H = self.hidden_dim
        C = self.num_classes
        i_osc = self.get_oscillation(1, T, seed=123)

        v = np.zeros((1, H), dtype=np.float32)
        a = np.zeros((1, H), dtype=np.float32)
        z = np.zeros((1, H), dtype=np.float32)
        y_out = np.zeros((1, C), dtype=np.float32)

        eps_v_rec = np.zeros((1, H), dtype=np.float32)
        eps_a_rec = np.zeros((1, H, H), dtype=np.float32)
        ebar_rec = np.zeros((1, H, H), dtype=np.float32)
        beta_row = self.beta_vec.reshape(1, H)
        beta_col = self.beta_vec.reshape(1, H, 1)
        has_alif = "alif" in self.cell_type

        hidden_spikes = []
        v_mean_trace = []
        thr_mean_trace = []
        osc_trace = []
        elig_trace = []
        probs_trace = []

        v_ca1_pos = 0.0
        v_ca1_neg = 0.0
        ca1_pos_spikes = []
        ca1_neg_spikes = []

        recall_logits_sum = np.zeros((1, C), dtype=np.float32)

        for t in range(T):
            z_prev = z
            x_t = x_single[:, t, :]

            eps_v_rec = self.alpha * eps_v_rec + z_prev
            thr_t = self.v_thresh + beta_row * a
            i_syn = x_t @ self.w_in.T + z_prev @ self.w_rec.T + i_osc[:, t, :]
            v = self.alpha * v + i_syn - z_prev * self.v_thresh

            v_scaled = (v - thr_t) / self.v_thresh
            z = (v_scaled >= 0.0).astype(np.float32)
            psi = (self.gamma_pd / self.v_thresh) * np.maximum(0.0, 1.0 - np.abs(v_scaled))
            psi_col = psi[:, :, None]
            ev_rec_3d = eps_v_rec[:, None, :]

            if has_alif:
                e_rec = psi_col * (ev_rec_3d - beta_col * eps_a_rec)
                eps_a_rec = psi_col * ev_rec_3d + (self.rho - psi_col * beta_col) * eps_a_rec
                a = self.rho * a + z
            else:
                e_rec = psi_col * ev_rec_3d
            ebar_rec = self.kappa * ebar_rec + e_rec

            y_out = self.kappa * y_out + (z @ self.w_out.T + self.b_out)
            pi_t = softmax_np(y_out)[0]

            if t >= recall_start:
                recall_logits_sum += y_out
                pred_prob_true = float(pi_t[y_single])
                ec_ground_truth = 0.0 if omit_target else 1.0
                signed_err = ec_ground_truth - pred_prob_true

                v_ca1_pos = 0.80 * v_ca1_pos + max(0.0, signed_err) * 0.62
                if v_ca1_pos >= 0.42:
                    ca1_pos_spikes.append(t)
                    v_ca1_pos -= 0.42

                v_ca1_neg = 0.80 * v_ca1_neg + max(0.0, -signed_err) * 0.62
                if v_ca1_neg >= 0.42:
                    ca1_neg_spikes.append(t)
                    v_ca1_neg -= 0.42

            hidden_spikes.append(z[0].astype(int).tolist())
            v_mean_trace.append(round(float(np.mean(v[0, :8])), 4))
            thr_mean_trace.append(round(float(np.mean(thr_t[0, :8])), 4))
            osc_trace.append(round(float(i_osc[0, t, 0]), 4))
            elig_trace.append(round(float(np.mean(np.abs(ebar_rec))) * 100.0, 4))
            probs_trace.append([round(float(p), 4) for p in pi_t])

        pred_class = int(np.argmax(recall_logits_sum, axis=-1)[0])
        hidden_np = np.array(hidden_spikes, dtype=np.float32)
        firing_rate_hz = float(np.sum(hidden_np)) / (self.hidden_dim * T * self.dt)
        synops = int(np.sum(hidden_np) * self.hidden_dim + float(np.sum(x_single)) * self.hidden_dim)

        # Count how many hidden spikes occurred specifically during the silent delay [cues_end, recall_start)
        delay_spikes = int(np.sum(hidden_np[cues_end:recall_start])) if recall_start > cues_end else 0

        return {
            "total_steps": int(T),
            "dt_ms": float(self.dt_ms),
            "cues_end_step": int(cues_end),
            "recall_start_step": int(recall_start),
            "true_label": int(y_single),
            "pred_label": int(pred_class),
            "is_correct": bool(pred_class == y_single),
            "firing_rate_hz": round(firing_rate_hz, 1),
            "synops": synops,
            "total_hidden_spikes": int(np.sum(hidden_np)),
            "delay_bridge_spikes": delay_spikes,
            "ca1_pos_miss_spikes": ca1_pos_spikes,
            "ca1_neg_violation_spikes": ca1_neg_spikes,
            "input_spikes": x_single[0].astype(int).tolist(),
            "hidden_spikes": hidden_spikes,
            "v_trace": v_mean_trace,
            "thr_trace": thr_mean_trace,
            "osc_trace": osc_trace,
            "elig_trace": elig_trace,
            "probs_trace": probs_trace,
        }


# ==============================================================================
# 3. GLOBAL WORKSPACE & SESSION STATE (Supports Both Local & Vercel Serverless)
# ==============================================================================
CURRENT_DATASET = {
    "dataset_type": "bellec_cue_accumulation",
    "delay_ms": 300,
    "cue_rate_hz": 80.0,
    "bg_rate_hz": 10.0,
    "batch_size": 36,
    "seed": 42,
    "generator": None,
    "probe_x": None,
    "probe_y": None,
    "recall_start": 0,
    "total_steps": 0,
    "cues_end": 0,
}

TRAINED_MODELS = {}
ACTIVE_SESSION = {
    "nets": {},
    "opts": {},
    "snapshots": {},
    "config": {},
}


def build_dataset(dataset_type="bellec_cue_accumulation", delay_ms=300,
                  cue_rate_hz=80.0, bg_rate_hz=10.0, batch_size=36, seed=42):
    with STATE_LOCK:
        if dataset_type == "delayed_cue_association":
            gen = DelayedCueAssociationDatasetGenerator(
                cue_rate_hz=cue_rate_hz, bg_rate_hz=bg_rate_hz
            )
        else:
            gen = CueAccumulationDatasetGenerator(
                cue_rate_hz=cue_rate_hz, bg_rate_hz=bg_rate_hz
            )

        x_probe, y_probe, r_start, t_steps, c_end = gen.sample_batch(
            batch_size=batch_size, delay_ms=float(delay_ms), seed=seed
        )
        CURRENT_DATASET.update({
            "dataset_type": dataset_type,
            "delay_ms": int(delay_ms),
            "cue_rate_hz": float(cue_rate_hz),
            "bg_rate_hz": float(bg_rate_hz),
            "batch_size": int(batch_size),
            "seed": int(seed),
            "generator": gen,
            "probe_x": x_probe,
            "probe_y": y_probe,
            "recall_start": int(r_start),
            "total_steps": int(t_steps),
            "cues_end": int(c_end),
        })

        sample_idx = 0
        in_spk = x_probe[sample_idx].astype(int).tolist()
        label_int = int(y_probe[sample_idx])
        return {
            "dataset_type": dataset_type,
            "num_classes": gen.num_classes,
            "class_names": gen.class_names,
            "input_dim": gen.input_dim,
            "dt_ms": gen.dt_ms,
            "delay_ms": int(delay_ms),
            "total_steps": int(t_steps),
            "total_duration_ms": int(t_steps * gen.dt_ms),
            "cues_end_step": int(c_end),
            "recall_start_step": int(r_start),
            "sample_label": label_int,
            "sample_label_name": gen.class_names[label_int],
            "sample_input_spikes": in_spk,
            "total_input_spikes": int(np.sum(in_spk)),
        }


build_dataset()


def _init_models_in_lock(model_ids, hidden_dim, lr, tau_mem_ms, tau_adapt_ms,
                         beta_adapt, f_theta, f_gamma, a_theta, omit_target):
    gen = CURRENT_DATASET["generator"]
    seed = CURRENT_DATASET["seed"]
    probe_x = CURRENT_DATASET["probe_x"][0:1].copy()
    probe_y = int(CURRENT_DATASET["probe_y"][0])
    eval_x = CURRENT_DATASET["probe_x"].copy()
    eval_y = CURRENT_DATASET["probe_y"].copy()
    r_start = CURRENT_DATASET["recall_start"]
    c_end = CURRENT_DATASET["cues_end"]

    nets = {}
    opts = {}
    snapshots = {mid: {} for mid in model_ids}
    init_models_payload = {}

    for idx, mid in enumerate(model_ids):
        cfg = MODEL_CATALOG[mid]
        net = VisualizerRecurrentSNN(
            input_dim=gen.input_dim,
            hidden_dim=hidden_dim,
            num_classes=gen.num_classes,
            cell_type=cfg["cell_type"],
            dt_ms=gen.dt_ms,
            tau_mem_ms=tau_mem_ms,
            tau_adapt_ms=tau_adapt_ms,
            beta_adapt=beta_adapt,
            f_theta=f_theta,
            f_gamma=f_gamma,
            a_theta=a_theta,
            seed=seed + idx * 7,
        )
        nets[mid] = net
        opts[mid] = NumpyAdamOptimizer(net.parameters(), lr=lr)

        _, init_acc, _ = net.forward_eprop(eval_x, eval_y, r_start, compute_grads=False, seed=seed)
        telem0 = net.record_single_trial_telemetry(probe_x, probe_y, r_start, c_end, omit_target=omit_target)
        snapshots[mid][0] = telem0
        TRAINED_MODELS[mid] = {"net": net, "snapshots": snapshots[mid], "epochs": 0}
        init_models_payload[mid] = {
            "meta": MODEL_CATALOG[mid],
            "acc": round(init_acc, 1),
            "telemetry": telem0,
        }

    ACTIVE_SESSION["nets"] = nets
    ACTIVE_SESSION["opts"] = opts
    ACTIVE_SESSION["snapshots"] = snapshots
    ACTIVE_SESSION["config"] = {
        "model_ids": model_ids,
        "hidden_dim": hidden_dim,
        "lr": lr,
        "tau_mem_ms": tau_mem_ms,
        "tau_adapt_ms": tau_adapt_ms,
        "beta_adapt": beta_adapt,
        "f_theta": f_theta,
        "f_gamma": f_gamma,
        "a_theta": a_theta,
        "omit_target": omit_target,
    }
    return init_models_payload


# ==============================================================================
# 4. FLASK ROUTES (Dataset Generator, Step-by-Step & SSE Training, Probe)
# ==============================================================================
@app.route("/favicon.ico")
def favicon():
    return ("", 204)


@app.route("/")
def index():
    return render_template("index.html", models=MODEL_CATALOG, device="Vectorized NumPy Engine")


@app.route("/api/generate_dataset", methods=["POST"])
def api_generate_dataset():
    data = request.get_json() or {}
    ds_type = str(data.get("dataset_type", "bellec_cue_accumulation"))
    delay_ms = int(np.clip(int(data.get("delay_ms", 300)), 50, 1000))
    cue_rate_hz = float(np.clip(float(data.get("cue_rate_hz", 80.0)), 30.0, 150.0))
    bg_rate_hz = float(np.clip(float(data.get("bg_rate_hz", 10.0)), 1.0, 40.0))
    batch_size = int(np.clip(int(data.get("batch_size", 36)), 16, 96))
    seed = int(data.get("seed", 42))

    info = build_dataset(
        dataset_type=ds_type,
        delay_ms=delay_ms,
        cue_rate_hz=cue_rate_hz,
        bg_rate_hz=bg_rate_hz,
        batch_size=batch_size,
        seed=seed,
    )
    return jsonify({"status": "success", "dataset": info})


@app.route("/api/init_train", methods=["POST"])
def api_init_train():
    """
    Initializes all selected models at Epoch 0 and returns initial untrained spike telemetry.
    Designed for Vercel-safe epoch-by-epoch JSON streaming (zero proxy buffering!).
    """
    data = request.get_json() or {}
    raw_models = data.get("models", ["lif_eprop", "lif_theta_eprop", "alif_theta_eprop"])
    if isinstance(raw_models, str):
        raw_models = [m.strip() for m in raw_models.split(",")]
    model_ids = [m for m in raw_models if m in MODEL_CATALOG] or ["alif_theta_eprop"]

    hidden_dim = int(np.clip(int(data.get("hidden_dim", 64)), 24, 96))
    lr = float(np.clip(float(data.get("lr", 0.0045)), 0.0005, 0.02))
    tau_mem_ms = float(np.clip(float(data.get("tau_mem_ms", 20.0)), 5.0, 60.0))
    tau_adapt_ms = float(np.clip(float(data.get("tau_adapt_ms", 500.0)), 100.0, 1200.0))
    beta_adapt = float(np.clip(float(data.get("beta_adapt", 1.6)), 0.0, 3.5))
    f_theta = float(np.clip(float(data.get("f_theta", 8.0)), 0.0, 20.0))
    f_gamma = float(np.clip(float(data.get("f_gamma", 40.0)), 15.0, 80.0))
    a_theta = float(np.clip(float(data.get("a_theta", 0.15)), 0.0, 0.40))
    omit_target = bool(data.get("omit_target", False))

    with STATE_LOCK:
        init_models = _init_models_in_lock(
            model_ids, hidden_dim, lr, tau_mem_ms, tau_adapt_ms,
            beta_adapt, f_theta, f_gamma, a_theta, omit_target
        )
    return jsonify({"status": "success", "epoch": 0, "models": init_models})


@app.route("/api/step_train", methods=["POST"])
def api_step_train():
    """
    Executes 1 training epoch across all active models and returns live telemetry immediately.
    Works 100% reliably on both local Flask and Vercel Serverless without SSE buffering.
    """
    data = request.get_json() or {}
    ep = int(np.clip(int(data.get("epoch", 1)), 1, 100))
    steps_per_epoch = int(np.clip(int(data.get("steps_per_epoch", 2)), 1, 6))
    omit_target = bool(data.get("omit_target", False))

    with STATE_LOCK:
        if not ACTIVE_SESSION["nets"]:
            _init_models_in_lock(
                ["lif_eprop", "lif_theta_eprop", "alif_theta_eprop"],
                64, 0.0045, 20.0, 500.0, 1.6, 8.0, 40.0, 0.15, omit_target
            )

        gen = CURRENT_DATASET["generator"]
        delay_ms = CURRENT_DATASET["delay_ms"]
        batch_size = CURRENT_DATASET["batch_size"]
        seed = CURRENT_DATASET["seed"]
        probe_x = CURRENT_DATASET["probe_x"][0:1].copy()
        probe_y = int(CURRENT_DATASET["probe_y"][0])
        eval_x = CURRENT_DATASET["probe_x"].copy()
        eval_y = CURRENT_DATASET["probe_y"].copy()
        r_start = CURRENT_DATASET["recall_start"]
        c_end = CURRENT_DATASET["cues_end"]

        nets = ACTIVE_SESSION["nets"]
        opts = ACTIVE_SESSION["opts"]
        snapshots = ACTIVE_SESSION["snapshots"]
        ep_models = {}

        for mid, net in nets.items():
            cfg = MODEL_CATALOG[mid]
            opt = opts[mid]
            losses = []

            for step_i in range(steps_per_epoch):
                step_seed = seed + ep * 100 + step_i * 13
                xb, yb, r_st, _, _ = gen.sample_batch(
                    batch_size=batch_size,
                    delay_ms=float(delay_ms),
                    seed=step_seed,
                )
                if cfg["rule"] == "bptt":
                    loss_val, _, grads = net.forward_bptt(xb, yb, r_st, seed=step_seed)
                else:
                    loss_val, _, grads = net.forward_eprop(xb, yb, r_st, compute_grads=True, seed=step_seed)
                opt.step(grads, max_norm=2.0)
                np.fill_diagonal(net.w_rec, 0.0)
                losses.append(loss_val)

            eval_loss, eval_acc, _ = net.forward_eprop(eval_x, eval_y, r_start, compute_grads=False, seed=seed + ep)
            telem = net.record_single_trial_telemetry(probe_x, probe_y, r_start, c_end, omit_target=omit_target)
            snapshots[mid][ep] = telem
            TRAINED_MODELS[mid] = {"net": net, "snapshots": snapshots[mid], "epochs": ep}

            ep_models[mid] = {
                "loss": round(float(np.mean(losses)), 4),
                "eval_loss": round(float(eval_loss), 4),
                "acc": round(float(eval_acc), 1),
                "telemetry": telem,
            }

    return jsonify({"status": "success", "epoch": ep, "models": ep_models})


@app.route("/api/stream_train", methods=["GET"])
def api_stream_train():
    """
    Server-Sent Events (SSE) endpoint kept for backwards compatibility and CLI streaming tests.
    """
    models_param = request.args.get("models", "alif_theta_eprop,lif_theta_eprop,lif_eprop")
    model_ids = [m.strip() for m in models_param.split(",") if m.strip() in MODEL_CATALOG] or ["alif_theta_eprop"]

    epochs = int(np.clip(int(request.args.get("epochs", 22)), 3, 60))
    steps_per_epoch = int(np.clip(int(request.args.get("steps_per_epoch", 2)), 1, 6))
    hidden_dim = int(np.clip(int(request.args.get("hidden_dim", 64)), 24, 96))
    lr = float(np.clip(float(request.args.get("lr", 0.0045)), 0.0005, 0.02))
    tau_mem_ms = float(np.clip(float(request.args.get("tau_mem_ms", 20.0)), 5.0, 60.0))
    tau_adapt_ms = float(np.clip(float(request.args.get("tau_adapt_ms", 500.0)), 100.0, 1200.0))
    beta_adapt = float(np.clip(float(request.args.get("beta_adapt", 1.6)), 0.0, 3.5))
    f_theta = float(np.clip(float(request.args.get("f_theta", 8.0)), 0.0, 20.0))
    f_gamma = float(np.clip(float(request.args.get("f_gamma", 40.0)), 15.0, 80.0))
    a_theta = float(np.clip(float(request.args.get("a_theta", 0.15)), 0.0, 0.40))
    omit_target = request.args.get("omit_target", "false").lower() == "true"

    def event_stream():
        with STATE_LOCK:
            init_models = _init_models_in_lock(
                model_ids, hidden_dim, lr, tau_mem_ms, tau_adapt_ms,
                beta_adapt, f_theta, f_gamma, a_theta, omit_target
            )
        yield f"data: {json.dumps({'event': 'init', 'epochs': epochs, 'models': init_models})}\n\n"

        t_start_all = time.perf_counter()
        for ep in range(1, epochs + 1):
            t_ep0 = time.perf_counter()
            with STATE_LOCK:
                gen = CURRENT_DATASET["generator"]
                delay_ms = CURRENT_DATASET["delay_ms"]
                batch_size = CURRENT_DATASET["batch_size"]
                seed = CURRENT_DATASET["seed"]
                probe_x = CURRENT_DATASET["probe_x"][0:1].copy()
                probe_y = int(CURRENT_DATASET["probe_y"][0])
                eval_x = CURRENT_DATASET["probe_x"].copy()
                eval_y = CURRENT_DATASET["probe_y"].copy()
                r_start = CURRENT_DATASET["recall_start"]
                c_end = CURRENT_DATASET["cues_end"]

                nets = ACTIVE_SESSION["nets"]
                opts = ACTIVE_SESSION["opts"]
                snapshots = ACTIVE_SESSION["snapshots"]
                ep_models = {}

                for mid in model_ids:
                    cfg = MODEL_CATALOG[mid]
                    net = nets[mid]
                    opt = opts[mid]
                    losses = []
                    for step_i in range(steps_per_epoch):
                        step_seed = seed + ep * 100 + step_i * 13
                        xb, yb, r_st, _, _ = gen.sample_batch(
                            batch_size=batch_size, delay_ms=float(delay_ms), seed=step_seed
                        )
                        if cfg["rule"] == "bptt":
                            loss_val, _, grads = net.forward_bptt(xb, yb, r_st, seed=step_seed)
                        else:
                            loss_val, _, grads = net.forward_eprop(xb, yb, r_st, compute_grads=True, seed=step_seed)
                        opt.step(grads, max_norm=2.0)
                        np.fill_diagonal(net.w_rec, 0.0)
                        losses.append(loss_val)

                    eval_loss, eval_acc, _ = net.forward_eprop(eval_x, eval_y, r_start, compute_grads=False, seed=seed + ep)
                    telem = net.record_single_trial_telemetry(probe_x, probe_y, r_start, c_end, omit_target=omit_target)
                    snapshots[mid][ep] = telem
                    TRAINED_MODELS[mid] = {"net": net, "snapshots": snapshots[mid], "epochs": ep}
                    ep_models[mid] = {
                        "loss": round(float(np.mean(losses)), 4),
                        "eval_loss": round(float(eval_loss), 4),
                        "acc": round(float(eval_acc), 1),
                        "telemetry": telem,
                    }

            ep_elapsed = time.perf_counter() - t_ep0
            if ep_elapsed < 0.20:
                time.sleep(0.20 - ep_elapsed)

            ep_data = {
                "event": "epoch",
                "epoch": ep,
                "total_epochs": epochs,
                "elapsed_sec": round(time.perf_counter() - t_start_all, 1),
                "models": ep_models,
            }
            yield f"data: {json.dumps(ep_data)}\n\n"

        yield f"data: {json.dumps({'event': 'done', 'elapsed_sec': round(time.perf_counter() - t_start_all, 1)})}\n\n"

    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")


@app.route("/api/probe_trial", methods=["POST"])
def api_probe_trial():
    data = request.get_json() or {}
    model_id = str(data.get("model_id", "alif_theta_eprop"))
    sample_idx = int(np.clip(int(data.get("sample_idx", 0)), 0, 15))
    omit_target = bool(data.get("omit_target", False))
    epoch_snap = data.get("epoch_snapshot", None)

    with STATE_LOCK:
        if model_id in TRAINED_MODELS:
            entry = TRAINED_MODELS[model_id]
            if epoch_snap is not None and int(epoch_snap) in entry["snapshots"] and not omit_target and sample_idx == 0:
                return jsonify({
                    "status": "success",
                    "model_id": model_id,
                    "epoch": int(epoch_snap),
                    "telemetry": entry["snapshots"][int(epoch_snap)],
                })
            net = entry["net"]
        else:
            cfg = MODEL_CATALOG.get(model_id, MODEL_CATALOG["alif_theta_eprop"])
            gen = CURRENT_DATASET["generator"]
            net = VisualizerRecurrentSNN(
                input_dim=gen.input_dim,
                hidden_dim=64,
                num_classes=gen.num_classes,
                cell_type=cfg["cell_type"],
                dt_ms=gen.dt_ms,
            )

        probe_x = CURRENT_DATASET["probe_x"][sample_idx : sample_idx + 1].copy()
        probe_y = int(CURRENT_DATASET["probe_y"][sample_idx])
        r_start = CURRENT_DATASET["recall_start"]
        c_end = CURRENT_DATASET["cues_end"]
        telem = net.record_single_trial_telemetry(probe_x, probe_y, r_start, c_end, omit_target=omit_target)

    return jsonify({
        "status": "success",
        "model_id": model_id,
        "telemetry": telem,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print("\n" + "=" * 80)
    print("URD-302 INTERACTIVE SNN MODEL VISUALIZER & LIVE FINE-TUNING STUDIO")
    print("=" * 80)
    print(f"[*] Student : Ko Sett Paing (KOSETTPAI) | Advisor: Dr. Etienne Mueller")
    print(f"[*] Engine  : Vectorized NumPy Online e-prop & Surrogate BPTT Engine")
    print(f"[*] Models  : 6 SNN Architectures (LIF, LIF+Theta, ALIF, ALIF+Theta, BPTT, Random-Phase)")
    print(f"[*] URL     : http://127.0.0.1:{port}")
    print("=" * 80 + "\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
