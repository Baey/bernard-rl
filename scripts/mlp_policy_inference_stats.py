import torch
import torch.nn as nn
import time
import numpy as np


class ActorMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(35, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )

    def forward(self, x):
        return self.model(x)


def load_fp32_model(weights_path='policy.pt', device='cpu'):
    model = ActorMLP().to(device)
    state = torch.load(weights_path, map_location=device)
    mapped = {}
    for k, v in state['model_state_dict'].items():
        if 'critic' in k or 'std' in k:
            continue
        mapped[k.replace('actor', 'model')] = v
    model.load_state_dict(mapped, strict=True)
    model.eval()
    return model


if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cuda"
    model = load_fp32_model(weights_path='/home/blaszar/bernard_rl/scripts/rsl_rl/logs/rsl_rl/bernard_locomotion/2025-09-10_11-25-57/model_6100.pt', device=device)
    model.eval()

    n_iter = 1000
    times = []

    with torch.no_grad():
        for i in range(n_iter):
            x = torch.rand(1, 35, device=device)  # losowe wejście
            start = time.perf_counter()
            y = model(x)
            end = time.perf_counter()
            elapsed = end - start
            times.append(elapsed)
            if i % 100 == 0:
                print(f"[{i}] output: {y.cpu().numpy()} | inference: {elapsed*1000:.3f} ms")

    # zapis statystyk do pliku .npy
    stats = {
        "times": np.array(times, dtype=np.float32),
        "freqs": 1.0 / np.array(times, dtype=np.float32)
    }
    np.save("inference_stats.npy", stats)
    print("Inference stats saved to inference_stats.npy")
