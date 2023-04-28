import numpy as np
import torch
import os
from torchdata.datapipes.iter import Shuffler

class Trajectory:
    """A class representing a trajectory of particle positions and materials.

    Assumes no particles are created or destroyed after the first frame."""
    def __init__(self, positions, materials, n_materials):
        self.len, self.n_particles, self.dim = positions.shape

        assert materials.shape[0] == self.n_particles
        assert materials.min() >= 0
        assert materials.max() < n_materials
        
        self.positions = positions
        self.materials = materials
        self.n_materials = n_materials

    def save(self, path):
        np.savez_compressed(path, positions=self.positions, materials=self.materials, n_materials=self.n_materials)

    @classmethod
    def load(cls, path):
        traj = np.load(path)
        return cls(traj['positions'], traj['materials'], traj['n_materials'].item())

    def get_datapoints(self, n_previous_velocities, device=None):
        """Return a list of Datapoints consisting of position, `n_previous_velocities`
        velocities, acceleration, and material for each particle."""
        assert n_previous_velocities >= 0

        # Compute velocities and accelerations.
        velocities = np.zeros_like(self.positions)
        velocities[1:] = self.positions[1:] - self.positions[:-1] # v_t = p_t - p_{t-1}
        accelerations = np.zeros_like(self.positions)
        accelerations[1:-1] = velocities[2:] - velocities[1:-1] # a_t = v_{t+1} - v_t

        points = []
        for frame in range(n_previous_velocities, self.len-1):
            pos = self.positions[frame]
            vel = np.concatenate([velocities[i] for i in range(frame-n_previous_velocities+1, frame+1)], axis=1)
            vel = vel.reshape((self.n_particles, n_previous_velocities, self.dim))
            acc = accelerations[frame]
            mat = self.materials

            points.append(TorchDatapoint(pos, vel, acc, mat, device=device))
        return points

class TorchDatapoint:
    """A class representing a single datapoint consisting of position, velocity,
    acceleration, and material for a single frame of a trajectory, all as torch
    tensors.
    """
    def __init__(self, positions, velocities, accelerations, materials,
                 device=None, _move=False):
        if _move:
            # In this case, the arguments should all be torch tensors.
            self.positions = positions.to(device)
            self.velocities = velocities.to(device)
            if accelerations is None:
                self.accelerations = None
            else:
                self.accelerations = accelerations.to(device)
            self.materials = materials.to(device)
        else:
            # In this case, the arguments should all be numpy arrays.
            self.positions = torch.tensor(positions, dtype=torch.float32, device=device)
            self.velocities = torch.tensor(velocities, dtype=torch.float32, device=device)
            if accelerations is None:
                self.accelerations = None
            else:
                self.accelerations = torch.tensor(accelerations, dtype=torch.float32, device=device)
            self.materials = torch.tensor(materials, dtype=torch.int64, device=device)

    def to(self, device):
        return TorchDatapoint(self.positions, self.velocities, self.accelerations, self.materials,
                              device=device, _move=True)

class TorchDataset(torch.utils.data.IterableDataset):
    """An iterable dataset of TorchDatapoints, with shuffle buffer support so
    that the dataset can be (sort of) shuffled without loading all datapoints
    into memory."""
    def __init__(self, dataset_dir, split, n_previous_velocities=5,
                 start_traj_idx=0, end_traj_idx=None,
                 shuffle=False, shuffle_buffer_size=10000, device=None):
        assert split in ['training', 'validation', 'test']
        assert n_previous_velocities >= 0

        self.dataset_dir = dataset_dir
        self.split = split
        self.n_previous_velocities = n_previous_velocities
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
        self.len = sum(self.n_traj_datapoints[start_traj_idx:end_traj_idx])

    def __len__(self):
        return self.len

    def _iter_without_buffer(self):
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
            datapoints = traj.get_datapoints(self.n_previous_velocities, device=self.device)

            # Shuffle the datapoints if necessary.
            if self.shuffle:
                np.random.shuffle(datapoints)

            # Yield the datapoints.
            for datapoint in datapoints:
                yield datapoint

    def __iter__(self):
        # Wrap the iterator in a Shuffler if necessary, which provides the
        # shuffle buffer functionality.
        if self.shuffle:
            return Shuffler(self._iter_without_buffer(),
                            buffer_size=self.shuffle_buffer_size).__iter__()
        else:
            return self._iter_without_buffer()
