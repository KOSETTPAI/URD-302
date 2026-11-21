# 🧠 Nested Neural Oscillations for Predictive Coding: Interactive SNN Model Visualizer & Fine-Tuning Studio

**Researcher:** Ko Sett Paing (`KOSETTPAI`)  
**Principal Investigator / Advisor:** Dr. Etienne Mueller  
**Course:** URD-302 (Undergraduate Research & Design — CMKL University)

---

## 🎯 1. What Is This Demo? (The Core Story in Plain English)

Standard Spiking Neural Networks (Leaky Integrate-and-Fire, **LIF**) suffer from a fundamental **timescale dilemma**:
- Biological membrane voltages decay within **$20\text{ ms}$** ($\tau_m \approx 20\text{ ms}$).
- Real-world behavioral decisions require remembering sensory cues across **hundreds of milliseconds** ($\Delta T = 100\text{--}800\text{ ms}$) of silence.
- Traditional deep learning solves this with **Backpropagation Through Time (BPTT)**, which stores the entire history of every neuron across time ($O(N \cdot T)$ memory) — something biological brains and low-power neuromorphic chips cannot do.

This interactive studio demonstrates how **online local learning (`e-prop`, $O(1)$ memory in time)** bridges long silent delays using two biologically grounded mechanisms:
1. **Network-Level Oscillatory Phase Gating ($8\text{ Hz}$ Theta + $40\text{ Hz}$ Gamma rhythms):** Organizes spikes into rhythmic phase windows.
2. **Cellular Threshold Adaptation (`ALIF / LSNN`):** Maintains a slow adaptation trace ($\tau_a = 500\text{ ms}$) across silent delays.

---

## 🕹️ 2. How to Play the Demo (Step-by-Step Guide)

Even if it is your first time opening the studio, you can run a complete experiment in **4 simple steps**:

### Step 1 — Generate a Spiking Trial Dataset (Left Sidebar, Top Box)
1. Pick one of the two temporal memory benchmarks:
   - **Task 1: Sequential Evidence Accumulation (Bellec et al., 2020):** The network sees 5 sequential Left/Right sensory cues ($0\text{--}200\text{ ms}$), waits through a **Silent Delay ($\Delta T$)**, and must report whether Left or Right cues were in the majority.
   - **Task 2: Delayed Cue-Target Association & Hippocampal Comparator:** The network sees a class-specific cue ($A, B, \text{or } C$), waits through a **Silent Delay ($\Delta T$)**, and predicts the target identity at recall.
2. Drag the **Silent Delay ($\Delta T$)** slider (e.g., `100 ms` for a short delay or `400 ms` for a long delay) and click **"🎲 1. Generate New Trial Dataset"**.

### Step 2 — Choose SNN Models to Compare (Left Sidebar, Middle Box)
Check 1 to 6 models to train side-by-side:
- **M1: Standard LIF (Online e-prop)** — *Fast Trace Baseline* ($\tau_m = 20\text{ ms}$ only; struggles on long delays).
- **M2: LIF + Theta-Gamma Oscillation (Online e-prop)** — *8 Hz / 40 Hz Coherent Phase Gating*.
- **M3: Adaptive LIF / LSNN (Online e-prop)** — *Slow Threshold Adaptation* ($\tau_a = 500\text{ ms}$).
- **M4: ALIF + Theta-Gamma Hybrid (Online e-prop)** — *Combined Cellular + Network Synergy*.
- **M5: ALIF / LIF (Offline Surrogate BPTT Ceiling)** — *Unrolled BPTT Upper-Bound Baseline*.
- **M6: Control — Desynchronized Random Theta Phase** — *Randomized phase control proving phase coherence is causally necessary*.

> 💡 **Shortcut:** Click any of the **1-Click Scenarios** in the top header (`⚡ Scenario 1: 100ms Theta Gain`, `🧬 Scenario 2: 400ms Long Delay Bridge`, or `🏆 Scenario 3: Compare All 6 Models`) to automatically configure a benchmark comparison!

### Step 3 — Fine-Tune Parameters & Click "▶ 2. Run Live Fine-Tuning" (Left Sidebar, Bottom Box)
- Adjust **Training Epochs** (default `20` epochs takes ~15–20 seconds live), **Learning Rate $\eta$**, **Membrane $\tau_m$**, **Adaptation $\tau_a$**, or **Theta Frequency $f_\theta$**.
- Click **"▶ 2. Run Live Fine-Tuning (Watch Spikes Evolve)"**.

### Step 4 — How to Read the Results as It Runs
Watch the 4 synchronized panels on the right update live every epoch:

| Panel | What It Shows | What to Look For |
| :--- | :--- | :--- |
| **💡 Live Neural Activity Decoder (Top Banner)** | Plain-English live explanation | Explains what is happening at the current epoch in real time. |
| **📈 Panel B: Multi-Model Accuracy vs. Epoch** | Live accuracy curves ($\%$) for all checked models | Watch **M2 (LIF+Theta)** and **M4 (ALIF+Theta)** climb from $\sim 50\%$ chance to **$85\text{--}95\%$**, while **M1 (Standard LIF)** or **M6 (Random Phase)** lags behind on longer delays! |
| **⚡ Panel C: Single-Trial Spike Raster Timeline** | Divided into **① Sensory Cue Input**, **② Silent Memory Delay**, and **③ Recall & Error Check** | At **Epoch 0 (Untrained)**, look at the bottom row in Zone ③: you will see many **Red Miss Error Spikes (`CA1+`)**. As the model learns (Epoch $1 \to 20$), **the Red Error Spikes disappear** and **Blue Hidden Bridge Spikes** appear across Zone ②! |
| **⏪ Compare Learning Epochs Slider** | Time-machine slider between `Ep 0` and `Ep 20` | After training finishes, drag this slider back and forth between `Ep 0` and `Ep 20` to inspect the exact before-vs-after spike transformation. |
| **▶ Play Slow-Mo Animation & Panel A** | Millisecond-by-millisecond playback ($t = 0 \to T\text{ ms}$) | Click **▶ Play Slow-Mo Animation** to watch spikes travel from Input Channels $\to$ Recurrent Ring (with purple Theta wave halo) $\to$ Readout & Comparator nodes. |
| **🚨 Omission Probe Checkbox** | Expectation Violation Test | After training a model, check **🚨 Omission Probe**. This withholds the expected target during recall, causing **Amber Surprise / Violation Spikes (`CA1-`)** to burst in Panel C! |

---

## 💻 3. How to Run Locally

### Prerequisites
- Python 3.9+ with `flask` and `numpy`:
  ```bash
  pip install -r requirements.txt
  ```

### Launch Server
- **Windows 1-Click:** Double-click `run_visualizer.bat`
- **Terminal (Windows / macOS / Linux):**
  ```bash
  python app.py
  ```
- Open **`http://127.0.0.1:5050`** in your browser.

---

## ☁️ 4. How to Deploy on Vercel (1-Click Import)

This repository includes `vercel.json` and a lightweight `requirements.txt` (`flask`, `numpy`, `<25 MB` bundle size) engineered specifically for **Vercel Python Serverless Functions**:

1. Push this repository to [`https://github.com/KOSETTPAI/URD-302`](https://github.com/KOSETTPAI/URD-302).
2. Go to [vercel.com/new](https://vercel.com/new) and click **Import** next to `KOSETTPAI/URD-302`.
3. Leave all build settings at their defaults and click **Deploy**.
4. In ~20 seconds, your live interactive SNN visualizer will be online!
