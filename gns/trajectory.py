import numpy as np
from scipy.spatial import KDTree

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

    def datapoints(self, n_previous_velocities, connectivity_radius):
        """Return a list of Datapoints consisting of position, `n_previous_velocities`
        velocities, acceleration, and material for each particle, as well as index
        tuples for particle pairs within `connectivity_radius` of each other."""
        assert n_previous_velocities >= 0
        assert connectivity_radius >= 0

        # Compute velocities and accelerations.
        velocities = np.zeros_like(self.positions)
        velocities[1:] = self.positions[1:] - self.positions[:-1] # v_t = p_t - p_{t-1}
        accelerations = np.zeros_like(self.positions)
        accelerations[1:-1] = velocities[2:] - velocities[1:-1] # a_t = v_{t+1} - v_t

        points = []
        for frame in range(n_previous_velocities, self.len-1):
            # Get node data.
            pos = self.positions[frame]
            vel = np.concatenate([velocities[frame-i] for i in range(n_previous_velocities-1, -1, -1)], axis=1)
            vel = vel.reshape((self.n_particles, n_previous_velocities, self.dim))
            acc = accelerations[frame]
            mat = self.materials

            # Get edge data.
            kdtree = KDTree(pos)
            neighbors = kdtree.query_ball_point(pos, connectivity_radius)
            neighbor_idxs = np.array([(i, j) for i in range(self.n_particles) for j in neighbors[i] if i != j])

            points.append(Datapoint(pos, vel, acc, mat, neighbor_idxs))
        return points

class Datapoint:
    def __init__(self, positions, velocities, accelerations, materials, neighbor_idxs):
        self.positions = positions
        self.velocities = velocities
        self.accelerations = accelerations
        self.materials = materials
        self.neighbor_idxs = neighbor_idxs