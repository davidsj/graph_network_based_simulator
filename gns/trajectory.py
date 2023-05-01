import numpy as np
import torch
import os
from torchdata.datapipes.iter import Batcher, Shuffler

class Trajectory:
    """A class representing a trajectory of particle positions and materials.

    Assumes no particles are created or destroyed after the first frame."""
    def __init__(self, positions, materials):
        self.len, self.n_particles, self.dim = positions.shape

        assert materials.shape[0] == self.n_particles
        assert materials.min() >= 0
        
        self.positions = positions
        self.materials = materials

    def save(self, path):
        np.savez_compressed(path, positions=self.positions, materials=self.materials)

    @classmethod
    def load(cls, path):
        traj = np.load(path)
        return cls(traj['positions'], traj['materials'])

    def get_datapoints(self, n_previous_velocities, cum_vel_noise_std=0.0, device=None):
        """Return a list of TorchDatapoints consisting of material, position,
        `n_previous_velocities` velocities, and target acceleration for each particle."""
        assert n_previous_velocities >= 0
        assert cum_vel_noise_std >= 0.0

        materials = torch.tensor(self.materials, dtype=torch.int64, device=device)
        positions = torch.tensor(self.positions, dtype=torch.float32, device=device)

        # Compute velocities and target accelerations.
        velocities = torch.zeros_like(positions)
        velocities[1:] = positions[1:] - positions[:-1] # v_t = p_t - p_{t-1}
        target_accelerations = torch.zeros_like(positions)
        target_accelerations[1:-1] = velocities[2:] - velocities[1:-1] # a_t = v_{t+1} - v_t

        if cum_vel_noise_std > 0.0:
            # Add noise to velocities and positions.
            n_noise_steps = velocities.shape[0] - 1
            noise_step_std = cum_vel_noise_std / np.sqrt(n_noise_steps)
            vel_noise = torch.randn_like(velocities) * noise_step_std
            vel_noise[0] = 0.0

            cum_vel_noise = torch.cumsum(vel_noise, dim=0)
            cum_pos_noise = torch.cumsum(cum_vel_noise, dim=0)
            velocities += cum_vel_noise
            positions += cum_pos_noise

            # Adjust target accelerations to remove the accumulated noise.
            target_accelerations[1:-1] -= cum_vel_noise[1:-1]

        points = []
        for frame in range(n_previous_velocities, self.len-1):
            pos = positions[frame]
            vel = velocities[frame-n_previous_velocities+1:frame+1].transpose(0, 1)
            acc = target_accelerations[frame]

            points.append(TorchDatapoint(materials, pos, vel, acc, device=device))
        return points

class TorchDatapoint:
    """A class representing a single datapoint consisting of material, position,
    velocity, and target acceleration for a single frame of a trajectory, all as torch
    tensors.

    The position and velocity potentially contain accumulated noise, in which case the
    target acceleration includes a component necessary to eliminate the noise
    accumulated up to that frame, so that the model learns to be robust to accumulated
    rollout error."""
    def __init__(self, materials, positions, velocities, target_accelerations=None, device=None):
        self.materials = materials.to(device)
        self.positions = positions.to(device)
        self.velocities = velocities.to(device)
        if target_accelerations is None:
            self.target_accelerations = None
        else:
            self.target_accelerations = target_accelerations.to(device)

    def to(self, device):
        return TorchDatapoint(self.materials, self.positions, self.velocities,
                              self.target_accelerations, device=device)

class TorchDataset(torch.utils.data.IterableDataset):
    """An iterable dataset of TorchDatapoints, with shuffle buffer support so
    that the dataset can be (sort of) shuffled without loading all datapoints
    into memory."""
    def __init__(self, dataset_dir, split,
                 n_previous_velocities=5, cum_vel_noise_std=0.0,
                 start_traj_idx=0, end_traj_idx=None, batch_size=1,
                 shuffle=False, shuffle_buffer_size=10000, device=None):
        assert split in ['training', 'validation', 'test']
        assert n_previous_velocities >= 0
        assert cum_vel_noise_std >= 0.0
        if split != 'training':
            assert cum_vel_noise_std == 0.0
        assert batch_size >= 1

        self.dataset_dir = dataset_dir
        self.split = split
        self.cum_vel_noise_std = cum_vel_noise_std
        self.n_previous_velocities = n_previous_velocities
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.device = device

        # Determine number of trajectories and datapoints.
        with open(os.path.join(dataset_dir, split, 'lengths'), 'r') as f:
            traj_lengths = [int(l) for l in f.read().splitlines()]
        self.start_traj_idx = start_traj_idx
        if end_traj_idx is None:
            self.end_traj_idx = len(traj_lengths)
        else:
            self.end_traj_idx = min(end_traj_idx, len(traj_lengths))

        self.n_traj_datapoints = [max(0, l - n_previous_velocities - 1) for l in traj_lengths]
        self.len = int(np.ceil(sum(self.n_traj_datapoints[start_traj_idx:end_traj_idx])
                               / batch_size))

    def __len__(self):
        return self.len

    def _base_iter(self):
        # Determine the order in which to iterate over trajectories.
        traj_idxs = list(range(self.start_traj_idx, self.end_traj_idx))
        if self.shuffle:
            np.random.shuffle(traj_idxs)

        for traj_idx in traj_idxs:
            if self.n_traj_datapoints[traj_idx] == 0:
                continue

            # Load the datapoints from this trajectory.
            traj_path = os.path.join(self.dataset_dir, self.split, f'{traj_idx}.npz')
            traj = Trajectory.load(traj_path)
            datapoints = traj.get_datapoints(self.n_previous_velocities, self.cum_vel_noise_std,
                                             device=self.device)

            # Shuffle the datapoints if necessary.
            if self.shuffle:
                np.random.shuffle(datapoints)

            # Yield the datapoints.
            for datapoint in datapoints:
                yield datapoint

    def __iter__(self):
        # Wrap the iterator in a Shuffler if necessary, which provides the
        # shuffle buffer functionality.
        #
        # Then wrap it in a Batcher.
        iter = self._base_iter()
        if self.shuffle:
            iter = Shuffler(iter, buffer_size=self.shuffle_buffer_size).__iter__()
        iter = Batcher(iter, batch_size=self.batch_size).__iter__()
        return iter
