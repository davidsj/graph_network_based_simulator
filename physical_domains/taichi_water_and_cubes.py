import os
import taichi as ti
import numpy as np
from taichi_elements.engine.mpm_solver import MPMSolver
import argparse
import json
import io
import contextlib

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gns

# Parse arguments.
parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', type=str, default='datasets/taichi_water_and_cubes')
parser.add_argument('--traj_len', type=int, default=1000)
parser.add_argument('--n_training', type=int, default=1000)
parser.add_argument('--n_validation', type=int, default=30)
parser.add_argument('--n_test', type=int, default=30)
parser.add_argument('--dt_ms', type=float, default=2.5)
parser.add_argument('--n_cubes', type=int, default=2)
parser.add_argument('--cube_size', type=float, default=0.1)
parser.add_argument('--water_size', type=float, default=0.4)
args = parser.parse_args()

# Validate bounds on arguments.
assert args.traj_len >= 0
assert args.n_training >= 0
assert args.n_validation >= 0
assert args.n_test >= 0
assert args.dt_ms > 0
assert args.n_cubes >= 0
assert args.cube_size > 0
assert args.water_size > 0

# Make sure cubes and water fit horizontally across.
assert args.cube_size * args.n_cubes <= 1.0
assert args.water_size <= 1.0
# Make sure cubes plus water fit vertically.
assert args.water_size + args.cube_size <= 1.0

# Make sure the output dir doesn't already exist and then create it.
assert not os.path.exists(args.out_dir)
os.makedirs(args.out_dir)

res = 64
# This is implicitly how the MPMSolver sets the boundaries.
box_boundaries = np.array([[-1/res, 1.0],
                            [-1/res, 1.0]])

# Write metadata as JSON.
with open(os.path.join(args.out_dir, 'metadata.json'), 'w') as f:
    md = vars(args)
    md['name'] = 'taichi_water_and_cubes'
    md['dimensions'] = 2
    md['n_materials'] = len(MPMSolver.materials)
    md['box_boundaries'] = box_boundaries.tolist()
    md['default_connectivity_radius'] = 0.015
    json.dump(md, f)

# Create the trajectories for each dataset split.
for split in ['training', 'validation', 'test']:
    n_trajs = args.__dict__[f'n_{split}']
    os.makedirs(os.path.join(args.out_dir, split))

    vel_sum = np.zeros(2)
    acc_sum = np.zeros(2)
    vel_sq_sum = np.zeros(2)
    acc_sq_sum = np.zeros(2)
    vel_n = 0
    acc_n = 0
    for traj_idx in range(n_trajs):
        positions = []

        ti.init(arch=ti.cuda)
        mpm = MPMSolver(res=(res, res), padding=0)

        # Create the water as high up as possible.
        min_left = 0.0
        max_left = 1.0 - args.water_size
        left = np.random.uniform(min_left, max_left)
        bottom = 1.0 - args.water_size
        mpm.add_cube(lower_corner=[left, bottom],
                    cube_size=[args.water_size, args.water_size],
                    material=MPMSolver.material_water)

        # Create the cubes below the water.
        for i in range(args.n_cubes):
            min_left = i/args.n_cubes
            max_left = (i+1)/args.n_cubes - args.cube_size
            left = np.random.uniform(min_left, max_left)

            min_bottom = 0.0
            max_bottom = 1.0 - args.water_size - args.cube_size
            bottom = np.random.uniform(min_bottom, max_bottom)

            mpm.add_cube(lower_corner=[left, bottom],
                        cube_size=[args.cube_size, args.cube_size],
                        material=MPMSolver.material_elastic)
        print(f'{split} trajectory {traj_idx:5d}: created {len(mpm.particle_info()["position"])} particles')

        # Simulate.
        for frame in range(args.traj_len):
            with io.StringIO() as buffer, contextlib.redirect_stdout(buffer):
                # We redirect mpm's stdout because it prints a lot of dots and
                # there's no way to turn that off.
                mpm.step(args.dt_ms / 1000.0)
            particles = mpm.particle_info()
            positions.append(particles['position'])
        materials = mpm.particle_info()['material']
        
        # Save the trajectory.
        traj = gns.Trajectory(np.array(positions), np.array(materials))
        traj.save(os.path.join(args.out_dir, split, f'{traj_idx}.npz'))

        # Compute statistics.
        vel = traj.positions[1:] - traj.positions[:-1]
        acc = vel[1:] - vel[:-1]
        vel_sum += vel.sum(axis=(0, 1))
        acc_sum += acc.sum(axis=(0, 1))
        vel_sq_sum += np.square(vel).sum(axis=(0, 1))
        acc_sq_sum += np.square(acc).sum(axis=(0, 1))
        vel_n += vel.size
        acc_n += acc.size

    # Write training statistics.
    if split == 'training':
        vel_mean = vel_sum/vel_n
        acc_mean = acc_sum/acc_n
        vel_std = np.sqrt(vel_sq_sum/vel_n - np.square(vel_mean))
        acc_std = np.sqrt(acc_sq_sum/acc_n - np.square(acc_mean))
        md['training_stats'] = {'vel': {'mean': vel_mean.tolist(), 'std': vel_std.tolist()},
                                'acc': {'mean': acc_mean.tolist(), 'std': acc_std.tolist()}}
        with open(os.path.join(args.out_dir, 'metadata.json'), 'w') as f:
            json.dump(md, f)

    with open(os.path.join(args.out_dir, split, 'lengths'), 'w') as f:
        f.write(f'{args.traj_len}\n'*n_trajs)
