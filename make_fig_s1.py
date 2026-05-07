"""Supplemental Figure S1: Two efficacy curves (A) + chronotype bar charts (B1/B2)."""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

os.makedirs("output", exist_ok=True)

panel_a  = np.array(Image.open("output/Two hypothetical efficacy curves.png"))
panel_b1 = np.array(Image.open("output/Evening_Optimal_Bar_Plot.png"))
panel_b2 = np.array(Image.open("output/Midday_Optimal_Bar_Plot.png"))

# Panel A source is 1:1 square; B panels are 8:5 (1.6:1).
# figsize=(12,7) + width_ratios=[1.3,1] sizes B rows to ~1.6:1 aspect and
# A column to ~1:1 — no squishing on either side.
fig = plt.figure(figsize=(12, 7))
gs = gridspec.GridSpec(2, 2, figure=fig,
                       width_ratios=[1.3, 1],
                       wspace=0.06, hspace=0.15)
plt.subplots_adjust(top=0.92, bottom=0.02, left=0.02, right=0.98)

ax_a  = fig.add_subplot(gs[:, 0])
ax_b1 = fig.add_subplot(gs[0, 1])
ax_b2 = fig.add_subplot(gs[1, 1])

ax_a.imshow(panel_a)
ax_b1.imshow(panel_b1)
ax_b2.imshow(panel_b2)

for ax in [ax_a, ax_b1, ax_b2]:
    ax.axis('off')

label_kw = dict(fontweight='bold', fontfamily='Arial', va='bottom', ha='left', clip_on=False)
ax_a.text( -0.01, 1.03, 'A',  transform=ax_a.transAxes,  fontsize=28, **label_kw)
ax_b1.text(-0.01, 1.03, 'B1', transform=ax_b1.transAxes, fontsize=24, **label_kw)
ax_b2.text(-0.01, 1.03, 'B2', transform=ax_b2.transAxes, fontsize=24, **label_kw)

evening_color = [0.6, 0.35, 0.4]
midday_color  = [0.0, 0.5,  0.0]
ax_b1.text(0.608, 1.03, 'Evening optimal', transform=ax_b1.transAxes,
           fontsize=14, fontweight='bold', fontfamily='Arial',
           color=evening_color, ha='center', va='bottom', clip_on=False)
ax_b2.text(0.608, 1.03, 'Midday optimal', transform=ax_b2.transAxes,
           fontsize=14, fontweight='bold', fontfamily='Arial',
           color=midday_color, ha='center', va='bottom', clip_on=False)

plt.savefig("output/fig_s1.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: output/fig_s1.png")
