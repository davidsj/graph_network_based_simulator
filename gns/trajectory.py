import numpy as np
import torch

class Trajectory:
    """A class representing a trajectory of particle positions and materials.

    Assumes no particles are created or destroyed after the first frame."""
    def __init__(self, positions, materials, n_materials):
        self.len = positions.shape[0]
        self.n_particles = positions.shape[1]
        self.dim = positions.shape[2]

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

    def get_datapoints(self, n_previous_velocities):
        """Return a list of Datapoints consisting of position, `n_previous_velocities`
        velocities, acceleration, and material for each particle."""

        # Compute velocities and accelerations.
        velocities = np.zeros_like(self.positions)
        velocities[1:] = self.positions[1:] - self.positions[:-1] # v_t = p_t - p_{t-1}
        accelerations = np.zeros_like(self.positions)
        accelerations[1:-1] = velocities[2:] - velocities[1:-1] # a_t = v_{t+1} - v_t

        points = []
        for frame in range(n_previous_velocities, self.len-1):
            pos = self.positions[frame]
            vel = np.concatenate([velocities[frame-i] for i in range(n_previous_velocities-1, -1, -1)], axis=1)
            vel = vel.reshape((self.n_particles, n_previous_velocities, self.dim))
            acc = accelerations[frame]
            mat = self.materials

            points.append(TorchDatapoint(pos, vel, acc, mat))
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
            self.accelerations = accelerations.to(device)
            self.materials = materials.to(device)
        else:
            # In this case, the arguments should all be numpy arrays.
            self.positions = torch.tensor(positions, dtype=torch.float32, device=device)
            self.velocities = torch.tensor(velocities, dtype=torch.float32, device=device)
            self.accelerations = torch.tensor(accelerations, dtype=torch.float32, device=device)
            self.materials = torch.tensor(materials, dtype=torch.int64, device=device)

    def to(self, device):
        return TorchDatapoint(self.positions, self.velocities, self.accelerations, self.materials,
                              device=device, _move=True)
