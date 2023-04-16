import numpy as np

class Trajectory:
    """A class representing a trajectory of particle positions and materials.

    Assumes no particles are created or destroyed after the first frame."""
    def __init__(self, positions, materials, n_materials):
        assert materials.shape[0] == positions.shape[1]
        assert materials.min() >= 0
        assert materials.max() < n_materials

        self.len = positions.shape[0]
        self.n_particles = positions.shape[1]
        self.dim = positions.shape[2]
        
        self.positions = positions
        self.materials = materials
        self.n_materials = n_materials

    def save(self, path):
        np.savez_compressed(path, positions=self.positions, materials=self.materials, n_materials=self.n_materials)

    @classmethod
    def load(cls, path):
        traj = np.load(path)
        return cls(traj['positions'], traj['materials'], traj['n_materials'])