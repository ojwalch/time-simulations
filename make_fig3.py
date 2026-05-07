"""Figure 3: Adherence bar charts (A/B/C) with observed-difference annotations."""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

os.makedirs("output", exist_ok=True)

panel_a = np.array(Image.open("output/Perfect adherence.png"))
panel_b = np.array(Image.open("output/Nonadherent to time.png"))
panel_c = np.array(Image.open("output/Nonadherent to time and drug.png"))

# figsize height chosen so 3 side-by-side panels are close to 4:3 aspect —
# no squishing, minimal letterbox.
fig = plt.figure(figsize=(15, 5))
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.1)
plt.subplots_adjust(top=0.90, bottom=0.14, left=0.02, right=0.98)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[0, 2])

ax_a.imshow(panel_a)
ax_b.imshow(panel_b)
ax_c.imshow(panel_c)

for ax in [ax_a, ax_b, ax_c]:
    ax.axis('off')

label_kw = dict(fontsize=26, fontweight='bold', fontfamily='Arial', va='bottom', ha='left', clip_on=False)
ax_a.text(-0.01, 1.03, 'A.', transform=ax_a.transAxes, **label_kw)
ax_b.text(-0.01, 1.03, 'B.', transform=ax_b.transAxes, **label_kw)
ax_c.text(-0.01, 1.03, 'C.', transform=ax_c.transAxes, **label_kw)

annot_kw = dict(fontsize=16, fontweight='bold', fontfamily='Arial', ha='center', va='top', clip_on=False)
annotations = [
    (ax_a, '20% observed\ndifference'),
    (ax_b, '9% observed\ndifference'),
    (ax_c, '7% observed\ndifference'),
]
for ax, text in annotations:
    ax.text(0.60, -0.04, text, transform=ax.transAxes, **annot_kw)

plt.savefig("output/fig3.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: output/fig3.png")
