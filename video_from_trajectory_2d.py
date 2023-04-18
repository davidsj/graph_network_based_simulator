import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import gns

# Parse arguments.
parser = argparse.ArgumentParser()
parser.add_argument('dataset_dir', type=str)
parser.add_argument('split', type=str)
parser.add_argument('traj_idx', type=int)
parser.add_argument('--out_file', type=str)
args = parser.parse_args()

# Validate arguments.
assert os.path.exists(args.dataset_dir)
assert args.split in ['training', 'validation', 'test']
assert args.traj_idx >= 0

# Load metadata.
with open(os.path.join(args.dataset_dir, 'metadata.json'), 'r') as f:
    md = json.load(f)
    assert md['dimensions'] == 2
    dataset_name = md['name']
    dt_ms = md['dt_ms']
    traj_len = md['traj_len']

# Check that the output file doesn't already exist.
if args.out_file is None:
    out_file = f'{dataset_name}_{args.split}_{args.traj_idx}.mp4'
else:
    out_file = args.out_file
assert out_file.endswith('.mp4')
assert not os.path.exists(out_file)

# Load the trajectory.
traj = gns.Trajectory.load(os.path.join(args.dataset_dir, args.split, f'{args.traj_idx}.npz'))
assert traj.len == traj_len
assert traj.dim == 2

# Set up the square figure and axis for animation.
fig, ax = plt.subplots(figsize=(6, 6))
ax.axis('off')
fig.tight_layout()

# Draw the bounding box.
if traj.box_boundaries is not None:
    (left, right), (bottom, top) = traj.box_boundaries
    ax.plot([left, right, right, left, left],
            [bottom, bottom, top, top, bottom],
            color='black', linewidth=1.0)

# Draw the particles.
particle_scatters = []
material_masks = []
for material in np.unique(traj.materials):
    material_mask = traj.materials == material
    particle_scatters.append(ax.scatter(traj.positions[0, material_mask, 0],
                                        traj.positions[0, material_mask, 1], s=0.5))
    material_masks.append(material_mask)

# Show the frame index.
frame_text = ax.text(0.1, 0.9, '', transform=ax.transAxes, fontsize=12)

# Animate the particles.
def update(frame):
    frame_text.set_text(f'Frame {frame:5d}, Time {frame*dt_ms:7.1f} ms')
    for material_mask, particle_scatter in zip(material_masks, particle_scatters):
        particle_scatter.set_offsets(traj.positions[frame, material_mask, :])
    return particle_scatters
ani = FuncAnimation(fig, update, frames=traj_len)

# Save the animation.
# 
# Note: VLC seems to play this correctly, but QuickTime Player plays with
# variable frame rate.
ani.save(out_file, fps=1000/dt_ms, extra_args=['-vcodec', 'libx264'])