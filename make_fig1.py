"""Figure 1: Prior dosing history — already fully assembled by main.py."""
import os
import shutil

os.makedirs("output", exist_ok=True)
shutil.copy("output/Prior dosing history.png", "output/fig1.png")
print("Saved: output/fig1.png")
