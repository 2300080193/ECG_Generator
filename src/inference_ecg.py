import torch
import matplotlib.pyplot as plt
from ecg_generator import Generator

noise_dim = 100

gen = Generator(noise_dim)
gen.load_state_dict(torch.load("checkpoints/generator.pth"))
gen.eval()

noise = torch.randn(1, noise_dim)

with torch.no_grad():
    ecg = gen(noise).numpy()[0]

plt.plot(ecg)
plt.title("Generated ECG (Final)")
plt.show()