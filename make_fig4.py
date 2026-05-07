"""Figure 4: PK kinetics example (A) + half-life comparison (B)."""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

os.makedirs("output", exist_ok=True)

panel_a = np.array(Image.open("output/1.5_08.png"))
panel_b = np.array(Image.open("output/Effect of half-life on time of day efficacy.png"))

fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(1, 2, figure=fig,
                       width_ratios=[1, 1],
                       wspace=0.05)
plt.subplots_adjust(top=0.88, bottom=0.02, left=0.02, right=0.98)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])

ax_a.imshow(panel_a)
ax_b.imshow(panel_b)

for ax in [ax_a, ax_b]:
    ax.axis('off')

label_kw = dict(fontsize=26, fontweight='bold', fontfamily='Arial', va='bottom', ha='left', clip_on=False)
ax_a.text(-0.01, 1.04, 'A', transform=ax_a.transAxes, **label_kw)
ax_b.text(-0.01, 1.04, 'B', transform=ax_b.transAxes, **label_kw)

ax_a.set_title('Dosing 8 hrs after target minimum',
               fontsize=14, fontfamily='Arial', pad=4)

plt.savefig("output/fig4.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: output/fig4.png")
