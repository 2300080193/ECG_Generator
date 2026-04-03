import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from ecg_data_loader import get_data
from preprocess_ecg import normalize
from ecg_generator import Generator
from ecg_critic import Critic
from wgan_ecg_model import gradient_penalty

# -------------------------------
# Load and preprocess data
# -------------------------------
X, _ = get_data()
X = normalize(X)
X = torch.tensor(X, dtype=torch.float32)

print("Final Shape:", X.shape)

# -------------------------------
# Models
# -------------------------------
gen = Generator()
critic = Critic()

# -------------------------------
# Optimizers
# -------------------------------
lr = 5e-5
gen_opt = optim.Adam(gen.parameters(), lr=lr, betas=(0.0, 0.9))
critic_opt = optim.Adam(critic.parameters(), lr=lr, betas=(0.0, 0.9))

# -------------------------------
# Training parameters
# -------------------------------
epochs = 50
batch_size = 64
noise_dim = 100
lambda_gp = 10
critic_iterations = 5

# -------------------------------
# Training loop
# -------------------------------
for epoch in range(epochs):

    # Shuffle data
    perm = torch.randperm(len(X))
    X = X[perm]

    for i in range(0, len(X), batch_size):
        real = X[i:i+batch_size]

        # -----------------------
        # Train Critic
        # -----------------------
        for _ in range(critic_iterations):
            noise = torch.randn(real.size(0), noise_dim)
            fake = gen(noise)

            critic_real = critic(real).mean()
            critic_fake = critic(fake).mean()

            gp = gradient_penalty(critic, real, fake)

            loss_critic = -(critic_real - critic_fake) + lambda_gp * gp

            critic_opt.zero_grad()
            loss_critic.backward()
            critic_opt.step()

        # -----------------------
        # Train Generator
        # -----------------------
        noise = torch.randn(real.size(0), noise_dim)
        fake = gen(noise)
        loss_gen = -critic(fake).mean()

        gen_opt.zero_grad()
        loss_gen.backward()
        gen_opt.step()

    print(f"Epoch {epoch+1}/{epochs} | Critic Loss: {loss_critic:.4f} | Gen Loss: {loss_gen:.4f}")

    # -------------------------------
    # Save intermediate outputs (FIXED)
    # -------------------------------
    if (epoch + 1) % 50 == 0:
        noise = torch.randn(1, noise_dim)

        gen.eval()  # switch to eval mode
        with torch.no_grad():
            sample = gen(noise).numpy()[0]
        gen.train()  # back to training mode

        plt.plot(sample)
        plt.title(f"Generated ECG - Epoch {epoch+1}")
        plt.savefig(f"output_epoch_{epoch+1}.png")
        plt.close()

# -------------------------------
# Final outputs (multiple samples)
# -------------------------------
gen.eval()
for i in range(5):
    noise = torch.randn(1, noise_dim)
    with torch.no_grad():
        generated = gen(noise).numpy()[0]

    plt.plot(generated)
    plt.title(f"Final ECG {i+1}")
    plt.savefig(f"final_ecg_{i+1}.png")
    plt.close()

# Show one final graph
noise = torch.randn(1, noise_dim)
with torch.no_grad():
    generated = gen(noise).numpy()[0]

plt.plot(generated)
plt.title("Final Generated ECG Signal")
plt.show()
import os
os.makedirs("checkpoints", exist_ok=True)

torch.save(gen.state_dict(), "checkpoints/generator.pth")
print("Model saved successfully")