"""Figure 2: Efficacy curve (A) + DLMO distributions (B1/B2) + bar charts (C1/C2)."""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

os.makedirs("output", exist_ok=True)

panel_a  = np.array(Image.open("output/Evening optimal efficacy curve.png"))
panel_b1 = np.array(Image.open("output/Homogeneous DLMO Distribution.png"))
panel_b2 = np.array(Image.open("output/Disrupted DLMO Distribution.png"))
panel_c1 = np.array(Image.open("output/Efficacy Homogeneous Evening Optimal.png"))
panel_c2 = np.array(Image.open("output/Efficacy Disrupted Evening Optimal.png"))

# width_ratios=[2,1,1] makes panel A twice as wide as each B/C column.
# Combined with figsize height chosen so B/C rows are 4:3, panel A content
# (also 4:3) fills its full allocated height — no squishing, no letterbox.
fig = plt.figure(figsize=(15, 6))
gs = gridspec.GridSpec(2, 3, figure=fig,
                       width_ratios=[2, 1, 1],
                       wspace=0.05, hspace=0.05)
plt.subplots_adjust(top=0.92, bottom=0.02, left=0.02, right=0.98)

ax_a  = fig.add_subplot(gs[:, 0])
ax_b1 = fig.add_subplot(gs[0, 1])
ax_b2 = fig.add_subplot(gs[1, 1])
ax_c1 = fig.add_subplot(gs[0, 2])
ax_c2 = fig.add_subplot(gs[1, 2])

ax_a.imshow(panel_a)
ax_b1.imshow(panel_b1)
ax_b2.imshow(panel_b2)
ax_c1.imshow(panel_c1)
ax_c2.imshow(panel_c2)

for ax in [ax_a, ax_b1, ax_b2, ax_c1, ax_c2]:
    ax.axis('off')

label_kw = dict(fontweight='bold', fontfamily='Arial', va='bottom', ha='left', clip_on=False)
ax_a.text( -0.01, 1.03, 'A',  transform=ax_a.transAxes,  fontsize=28, **label_kw)
ax_b1.text(-0.01, 1.03, 'B1', transform=ax_b1.transAxes, fontsize=24, **label_kw)
ax_b2.text(-0.01, 1.03, 'B2', transform=ax_b2.transAxes, fontsize=24, **label_kw)
ax_c1.text(-0.01, 1.03, 'C1', transform=ax_c1.transAxes, fontsize=24, **label_kw)
ax_c2.text(-0.01, 1.03, 'C2', transform=ax_c2.transAxes, fontsize=24, **label_kw)

plt.savefig("output/fig2.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: output/fig2.png")
