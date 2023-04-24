import numpy as np
import torch
from scipy.spatial import KDTree
from .trajectory import TorchDatapoint

class MLP(torch.nn.Module):
    """Shared MLP architecture for encoder, processor, and decoder."""
    def __init__(self, in_dim, out_dim, hidden_dim=128, n_hidden=2):
        super().__init__()
        assert n_hidden >= 1

        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(in_dim, hidden_dim))
        for _ in range(n_hidden - 1):
            self.layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(torch.nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = torch.nn.functional.relu(x)
        x = self.layers[-1](x)
        return x

class FeatureEncoder(torch.nn.Module):
    """Shared encoder architecture for node and edge features."""
    def __init__(self, in_dim, embedding_dim=128):
        super().__init__()
        self.mlp = MLP(in_dim, embedding_dim)
        self.layer_norm = torch.nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = self.mlp(x)
        x = self.layer_norm(x)
        return x

class Encoder(torch.nn.Module):
    """Encodes a datapoint into the appropriate graph representation,
    excluding acceleration (which is the prediction target)."""
    def __init__(self, n_materials, physical_dim, n_previous_velocities,
                 connectivity_radius, box_boundaries=None,
                 material_embedding_dim=16, node_embedding_dim=128, edge_embedding_dim=128):
        super().__init__()
        assert n_previous_velocities >= 0
        assert connectivity_radius >= 0

        self.n_materials = n_materials
        self.physical_dim = physical_dim
        self.n_previous_velocities = n_previous_velocities
        self.connectivity_radius = connectivity_radius
        self.box_boundaries = box_boundaries

        # Materials are embedded from a one-hot representation.
        self.material_encoder = torch.nn.Linear(n_materials, material_embedding_dim)

        # Nodes are encoded from a concatenation of the previous velocities,
        # material of the corresponding particle, and (if present) distance
        # to orthogonal walls, clipped to `connectivity_radius`.
        node_feature_dim = n_previous_velocities*physical_dim + material_embedding_dim
        if box_boundaries is not None:
            node_feature_dim += 2*physical_dim
        self.node_encoder = FeatureEncoder(node_feature_dim, node_embedding_dim)

        # Edges are encoded from a concatenation of the relative position and
        # distance between the two neighboring particles.
        self.edge_encoder = FeatureEncoder(physical_dim + 1, edge_embedding_dim)

    def forward(self, dp):
        if isinstance(dp, TorchDatapoint):
            dps = [dp]
        else:
            dps = dp
        # This currently assumes all datapoints in the batch have the
        # same number of particles. TODO: Pad with dummy particles.
        #
        # N = number of datapoints
        # P = number of particles
        # D = physical dimension
        # V = number of previous velocities
        mat = torch.stack([dp.materials for dp in dps])  # N x P
        pos = torch.stack([dp.positions for dp in dps])  # N x P x D
        vel = torch.stack([dp.velocities for dp in dps]) # N x P x V x D
        n_datapoints, n_particles = mat.shape
        device = mat.device

        # Make a one-hot representation of the materials, then embed them.
        materials = torch.nn.functional.one_hot(mat, self.n_materials).to(torch.float32)
        materials = self.material_encoder(materials)

        # Concatenate the velocities, materials, and (if present) distance
        # to orthogonal walls, clipped to `connectivity_radius`.
        velocities = vel.reshape(n_datapoints, n_particles, -1)
        node_features = torch.cat([velocities, materials], dim=-1)
        if self.box_boundaries is not None:
            wall_dist = torch.zeros(n_datapoints, n_particles, self.physical_dim, 2,
                                    device=device)
            for i in range(self.physical_dim):
                wall_dist[:, :, i, 0] = pos[:, :, i] - self.box_boundaries[i][0]
                wall_dist[:, :, i, 1] = self.box_boundaries[i][1] - pos[:, :, i]
            wall_dist = torch.clamp(wall_dist, min=0.0, max=self.connectivity_radius)
            wall_dist = wall_dist.reshape(n_datapoints, n_particles, -1)
            node_features = torch.cat([node_features, wall_dist], dim=-1)
        nodes = self.node_encoder(node_features)

        # Identify edges. TODO: Do on GPU or at least parallelize across the batch dimension?
        neighbor_idxs = []
        for i, dp in enumerate(dps):
            if self.connectivity_radius in dp.neighbor_idx_cache:
                dp_neighbor_idxs = dp.neighbor_idx_cache[self.connectivity_radius].to(device)
            else:
                pos_np = pos[i].detach().cpu().numpy()
                kdtree = KDTree(pos_np)
                neighbors = kdtree.query_ball_point(pos_np, self.connectivity_radius)
                dp_neighbor_idxs = torch.tensor([(j, k) for j in range(n_particles) for k in neighbors[j] if j != k],
                                            device=device, dtype=torch.int64)
                dp_neighbor_idxs = dp_neighbor_idxs.reshape((-1, 2)) # ensure the right shape in case there are no neighbors
                dp.neighbor_idx_cache[self.connectivity_radius] = dp_neighbor_idxs.to('cpu')
            datapoint_idx = torch.tensor([[i]], device=device).broadcast_to(dp_neighbor_idxs.shape[0], 1)
            dp_neighbor_idxs = torch.cat([datapoint_idx, dp_neighbor_idxs], dim=1)
            neighbor_idxs.append(dp_neighbor_idxs)
        neighbor_idxs = torch.cat(neighbor_idxs, dim=0) # (datapoint_idx, receiver_idx, sender_idx)

        # Compute the relative positions and distances.
        receiver_pos = pos[neighbor_idxs[:, 0], neighbor_idxs[:, 1]]
        sender_pos = pos[neighbor_idxs[:, 0], neighbor_idxs[:, 2]]
        relative_positions = receiver_pos - sender_pos
        distances = torch.norm(relative_positions, dim=1, keepdim=True)
        edge_features = torch.cat([relative_positions, distances], dim=1)
        edges = self.edge_encoder(edge_features)

        return nodes, edges, neighbor_idxs

class Processor(torch.nn.Module):
    """`n_layers` graph network blocks, as described in Battaglia et al (2018).
    Relational inductive biases, deep learning, and graph networks.

    Node and edge updates go through a residual stream. Reduction is by summation,
    and there is no global state."""
    def __init__(self, n_layers=10, node_embedding_dim=128, edge_embedding_dim=128):
        super().__init__()

        self.edge_updaters = torch.nn.ModuleList()
        self.edge_layer_norms = torch.nn.ModuleList()
        self.node_updaters = torch.nn.ModuleList()
        self.node_layer_norms = torch.nn.ModuleList()

        for _ in range(n_layers):
            self.edge_updaters.append(MLP(edge_embedding_dim + 2*node_embedding_dim, edge_embedding_dim))
            self.edge_layer_norms.append(torch.nn.LayerNorm(edge_embedding_dim))
            self.node_updaters.append(MLP(edge_embedding_dim + node_embedding_dim, node_embedding_dim))
            self.node_layer_norms.append(torch.nn.LayerNorm(node_embedding_dim))

    def forward(self, nodes, edges, neighbor_idxs):
        for edge_updater, edge_layer_norm, node_updater, node_layer_norm in zip(
            self.edge_updaters, self.edge_layer_norms, self.node_updaters, self.node_layer_norms):
            # Update the edges.
            receivers = nodes[neighbor_idxs[:, 0], neighbor_idxs[:, 1]]
            senders = nodes[neighbor_idxs[:, 0], neighbor_idxs[:, 2]]
            edges_out = edge_updater(torch.cat([edges, receivers, senders], dim=-1))
            edges_out = edge_layer_norm(edges_out)
            edges = edges + edges_out

            # Aggregate the edges for each node. We do this with nodes flattened
            # across the batch dimension so we can use scatter_add.
            aggregated_edges = torch.zeros(nodes.shape[0]*nodes.shape[1], edges.shape[-1],
                                           device=edges.device)
            index = neighbor_idxs[:, [0]]*nodes.shape[1] + neighbor_idxs[:, [1]]
            index = index.broadcast_to(edges.shape)
            aggregated_edges = aggregated_edges.scatter_add(0, index, edges)
            aggregated_edges = aggregated_edges.reshape(nodes.shape[0], nodes.shape[1], -1)

            # Update the nodes.
            nodes_out = node_updater(torch.cat([aggregated_edges, nodes], dim=-1))
            nodes_out = node_layer_norm(nodes_out)
            nodes = nodes + nodes_out

        return nodes, edges, neighbor_idxs

class Decoder(MLP):
    """Decodes graph node output into predicted acceleration."""
    def __init__(self, physical_dim, node_embedding_dim=128):
        super().__init__(node_embedding_dim, physical_dim)

class GNS(torch.nn.Module):
    """Graph network-based simulator for predicting particle acceleration,
    composed of an Encoder, Processor, and Decoder."""
    def __init__(self, n_materials, physical_dim, n_previous_velocities,
                 connectivity_radius, box_boundaries=None,
                 n_processor_layers=10):
        super().__init__()
        self.encoder = Encoder(n_materials, physical_dim, n_previous_velocities,
                               connectivity_radius, box_boundaries=box_boundaries)
        self.processor = Processor(n_layers=n_processor_layers)
        self.decoder = Decoder(physical_dim)

    def forward(self, dp):
        nodes, edges, neighbor_idxs = self.encoder(dp)
        nodes, edges, neighbor_idxs = self.processor(nodes, edges, neighbor_idxs)
        nodes = self.decoder(nodes)

        # The modules return data with the batch dimension, so we remove it
        # if that's not how it was given to us.
        if isinstance(dp, TorchDatapoint):
            return nodes[0]
        else:
            return nodes
