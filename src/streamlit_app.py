import streamlit as st
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Fix import path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from ecg_generator import Generator

# -----------------------------
# Load model
# -----------------------------
noise_dim = 100
model_path = "checkpoints/generator.pth"

gen = Generator(noise_dim)
gen.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
gen.eval()

# -----------------------------
# UI
# -----------------------------
st.title("ECG Signal Generator (WGAN)")

st.write("Click the button to generate a synthetic ECG signal")

# Number of samples
num_samples = st.slider("Number of ECGs", 1, 5, 1)

if st.button("Generate ECG"):
    for i in range(num_samples):
        noise = torch.randn(1, noise_dim)

        with torch.no_grad():
            generated = gen(noise).numpy()[0]

        # Plot
        fig, ax = plt.subplots()
        ax.plot(generated)
        ax.set_title(f"Generated ECG {i+1}")

        st.pyplot(fig)

        # Download option
        csv = np.array(generated)
        st.download_button(
            label=f"Download ECG {i+1} (CSV)",
            data=",".join(map(str, csv)),
            file_name=f"ecg_{i+1}.csv",
            mime="text/csv"
        )