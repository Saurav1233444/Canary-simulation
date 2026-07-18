"""
╔══════════════════════════════════════════════════════════════════╗
║   CANARY — MobileNetV2 Simulator + Bayesian Memory Monitor       ║
║                                                                  ║
║  Loads Training dataset, applies MobileNetV2-like NumPy          ║
║  dummy convolutions, accumulates X/Y arrays in RAM, and uses     ║
║  Bayesian Change-Point Detection (BCPD) to halt execution        ║
║  before a terminal crash occurs due to out-of-memory.            ║
╚══════════════════════════════════════════════════════════════════╝

RUN WITH:
    source backend/venv/bin/activate
    python train_mobilenet.py
"""

import os
import sys
import psutil
import subprocess
import numpy as np
import warnings
from PIL import Image
warnings.filterwarnings("ignore")

# ─── path setup so we can import models/bcpd.py ───────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from models.bcpd import BayesianChangePointDetection

# ─── Configuration ────────────────────────────────────────────────
IMG_SIZE           = (224, 224)
NUM_CLASSES        = 7
DATA_ROOT          = os.path.join(PROJECT_ROOT, "data", "Training")

# Bayesian memory guard settings
HAZARD_RATE        = 1 / 50          # sensitivity
CP_ALERT_THRESHOLD = 0.3             # change-point probability → halt

# ANSI colours
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


# ══════════════════════════════════════════════════════════════════
#  1. PURE NUMPY MOBILENET-V2 SIMULATOR
# ══════════════════════════════════════════════════════════════════

class NumPyMobileNetV2:
    """
    A lightweight, pure-NumPy simulator for MobileNetV2 inference.
    Instead of downloading hundreds of megabytes of ML framework binaries,
    this class simulates the structural operation and memory footprint
    of a MobileNetV2 forward pass (depthwise + pointwise convolutions)
    using numpy matrices.
    """
    def __init__(self, num_classes=7):
        print(f"\n{CYAN}{BOLD}[Canary] Initializing Pure NumPy MobileNetV2 Simulator...{RESET}")
        # Dummy weights for a final dense layer
        self.W_dense = np.random.randn(1280, num_classes).astype(np.float32)

    def process_batch(self, X_batch: np.ndarray) -> np.ndarray:
        """
        Simulate processing a batch of shape (B, 224, 224, 3)
        We mimic a feature extraction pass by allocating an intermediate
        activation tensor (B, 7, 7, 1280), which uses up memory naturally.
        """
        B = X_batch.shape[0]
        # Simulate final feature map spatial size of 7x7 with 1280 channels
        # Allocate float32 (uses ~ B * 7 * 7 * 1280 * 4 bytes)
        simulated_features = np.random.randn(B, 7, 7, 1280).astype(np.float32)

        # Global average pooling (B, 1280)
        gap = np.mean(simulated_features, axis=(1, 2))

        # Softmax head
        logits = gap @ self.W_dense
        exp_L = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_L / np.sum(exp_L, axis=1, keepdims=True)
        return probs


# ══════════════════════════════════════════════════════════════════
#  2. BAYESIAN MEMORY GUARD
# ══════════════════════════════════════════════════════════════════

class BayesianMemoryGuard:
    """
    Monitors process RAM after every batch via psutil.
    Feeds readings into BCPD (Bayesian Change-Point Detection) to calculate
    P(change-point | RAM Data). Halts training if it spikes.
    """
    def __init__(self, hazard_rate=HAZARD_RATE, threshold=CP_ALERT_THRESHOLD):
        self.bcpd      = BayesianChangePointDetection(hazard_rate=hazard_rate)
        self.threshold = threshold
        self.history   = []
        self._proc     = psutil.Process(os.getpid())

    def ram_mb(self) -> float:
        return self._proc.memory_info().rss / (1024 ** 2)

    def check(self, batch_idx: int) -> tuple:
        ram          = self.ram_mb()
        cp_prob, _   = self.bcpd.update(ram)
        is_alert     = (cp_prob > self.threshold)
        self.history.append((batch_idx, ram, cp_prob, is_alert))
        return ram, cp_prob, is_alert


# ══════════════════════════════════════════════════════════════════
#  3. TRAINING DATA ACCUMULATOR
# ══════════════════════════════════════════════════════════════════

def get_system_vram_or_ram_gb():
    """Returns the available VRAM (if nvidia-smi is present) or system RAM in GB."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], 
            stderr=subprocess.STDOUT
        ).decode("utf-8")
        vram_mb = int(output.strip().split('\n')[0])
        return vram_mb / 1024.0
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return psutil.virtual_memory().total / (1024 ** 3)

def predict_batch_size(memory_gb):
    """
    Predicts safe starting batch size based on memory:
    """
    if memory_gb <= 4.5:
        return 8
    elif memory_gb <= 6.5:
        return 16
    elif memory_gb <= 8.5:
        return 32
    else:
        return 32

def predict_epochs(dataset_size):
    """
    Predicts typical epochs based on dataset size:
    """
    if dataset_size < 10000:
        return 20
    elif dataset_size <= 100000:
        return 10
    else:
        return 5

def run_canary_training_loop():
    """
    Loads images chunk-by-chunk, runs MobileNetV2, and accumulates X
    inputs in memory to simulate a memory leak / infinite allocation risk.
    """
    guard = BayesianMemoryGuard()
    model = NumPyMobileNetV2(num_classes=NUM_CLASSES)

    # We will accumulate X and Y arrays in memory here
    X_accumulated = []
    Y_accumulated = []

    # Discover images
    image_paths = []
    for d in range(NUM_CLASSES):
        folder = os.path.join(DATA_ROOT, str(d))
        if os.path.isdir(folder):
            for f in os.listdir(folder):
                if f.lower().endswith(('.jpg', '.png')):
                    image_paths.append((os.path.join(folder, f), d))

    dataset_size = len(image_paths)
    if dataset_size == 0:
        print(f"{RED}[Canary] No images found. Exiting.{RESET}")
        return

    # Dynamic prediction
    memory_gb = get_system_vram_or_ram_gb()
    pred_batch_size = predict_batch_size(memory_gb)
    pred_epochs = predict_epochs(dataset_size)
    steps_per_epoch = max(1, dataset_size // pred_batch_size)

    print(f"\n{CYAN}{BOLD}--- Hardware & Dataset Analysis ---{RESET}")
    print(f"Detected Memory: {memory_gb:.2f} GB")
    print(f"Dataset Size:    {dataset_size} images")
    print(f"Predicted Batch Size: {pred_batch_size}")
    print(f"Predicted Epochs:     {pred_epochs}")
    print(f"Steps per epoch:      {steps_per_epoch}")

    while True:
        try:
            ans = input(f"\nRun with predicted values? [Y/n] or type 'manage' to increment configs: ").strip().lower()
            if ans in ['', 'y', 'yes']:
                final_batch_size = pred_batch_size
                final_epochs = pred_epochs
                break
            elif ans == 'manage':
                new_bs = input(f"Enter new Batch Size (current: {pred_batch_size}): ")
                new_bs = int(new_bs) if new_bs else pred_batch_size
                new_ep = input(f"Enter new Epochs (current: {pred_epochs}): ")
                new_ep = int(new_ep) if new_ep else pred_epochs
                final_batch_size = new_bs
                final_epochs = new_ep
                break
            elif ans in ['n', 'no']:
                print("Exiting.")
                return
        except ValueError:
            print("Invalid input, please try again.")

    print(f"\n{BOLD}{'Batch':>6} │ {'RAM (MB)':>10} │ {'CP Prob':>10} │ Status{RESET}")
    print("─" * 60)

    global_batch = 0
    halt = False

    # Process in epochs and batches
    for epoch in range(final_epochs):
        if halt:
            break
        print(f"\n{CYAN}--- Epoch {epoch+1}/{final_epochs} ---{RESET}")
        np.random.shuffle(image_paths)
        
        for i in range(0, dataset_size, final_batch_size):
            chunk = image_paths[i:i+final_batch_size]
            global_batch += 1

            X_batch_list = []
            Y_batch_list = []

            # Load and normalize batch
            for FilePath, label in chunk:
                try:
                    img = Image.open(FilePath).convert("RGB").resize(IMG_SIZE)
                    arr = np.array(img, dtype=np.float32) / 255.0  # normalize
                    X_batch_list.append(arr)
                    Y_batch_list.append(label)
                except Exception:
                    pass

            if not X_batch_list:
                continue

            X_batch = np.stack(X_batch_list, axis=0)
            Y_batch = np.array(Y_batch_list, dtype=np.int32)

            # Use all X and Y as input (Simulating MobileNetV2)
            _ = model.process_batch(X_batch)

            # Intentionally accumulate data locally to force RAM usage growth
            X_accumulated.append(X_batch)
            Y_accumulated.append(Y_batch)
            
            # Exponential memory allocation surge after batch 15
            # This simulates the "crashing regime" where memory suddenly balloons
            if global_batch > 15:
                surge_size = int(1.2 ** (global_batch - 15) * 5 * 1024 * 1024) # MBs of floats
                X_accumulated.append(np.ones(surge_size, dtype=np.float32))

            # Bayesian Monitor Check
            ram_mb, cp_prob, is_alert = guard.check(global_batch)

            if is_alert:
                status = f"{RED}{BOLD}⚠ ALERT ⚠{RESET}"
            elif cp_prob > guard.threshold * 0.5:
                status = f"{YELLOW}⚡ ELEVATED{RESET}"
            else:
                status = f"{GREEN}✓ OK{RESET}"

            print(f"{global_batch:>6} │ {ram_mb:>10.1f} │ {cp_prob:>10.4f} │ {status}")

            if is_alert:
                _fire_canary_warning(ram_mb, cp_prob, guard.threshold)
                halt = True
                break

    if not halt:
        print(f"\n{GREEN}[Canary] Processing completed without hitting alert threshold.{RESET}")
    print()


def _fire_canary_warning(ram_mb, cp_prob, threshold):
    print("\n" + "═" * 65)
    print(f"{RED}{BOLD}  🐦 CANARY EARLY WARNING FIRED{RESET}")
    print("═" * 65)
    print(f"  {YELLOW}Bayesian Change-Point detected in memory usage!{RESET}")
    print(f"\n  RAM at alert   : {ram_mb:.1f} MB")
    print(f"  CP Probability : {cp_prob:.4f}  (threshold = {threshold})")
    print(f"\n  How Bayesian theorem stopped the program:")
    print(f"  ───────────────────────────────────────")
    print(f"  At each step, BCPD computed P(change-point | RAM readings).")
    print(f"  The accumulation of loaded X and Y arrays caused the terminal's")
    print(f"  memory to shift into a steep growth regime.")
    print(f"  The Bayesian predictive model assigned a posterior > {threshold:.0%}")
    print(f"  to this new regime.")
    print(f"\n  {RED}{BOLD}► PROGRAM HALTED. Canary prevented the terminal from crashing!{RESET}")
    print("═" * 65 + "\n")


def main():
    print(f"\n{BOLD}{'═'*65}")
    print("  CANARY — Data Intake & MobileNetV2 Simulation")
    print(f"{'═'*65}{RESET}")
    print("  Mode        : Pure NumPy / SciPy (Framework Free)")
    print("  Objective   : Load dataset into X, Y and apply model simulation")
    print("  Protections : Bayesian Memory Guard ENABLED")

    try:
        run_canary_training_loop()
    except KeyboardInterrupt:
        print(f"\n{YELLOW}[Canary] Cancelled by user.{RESET}")


if __name__ == "__main__":
    main()
