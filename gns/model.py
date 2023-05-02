import torch
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
                 connectivity_radius, vel_stats, cum_vel_noise_std,
                 box_boundaries=None,
                 material_embedding_dim=16, node_embedding_dim=128, edge_embedding_dim=128,
                 self_edges=True):
        super().__init__()
        assert n_previous_velocities >= 0
        assert connectivity_radius >= 0

        self.n_materials = n_materials
        self.physical_dim = physical_dim
        self.n_previous_velocities = n_previous_velocities
        self.connectivity_radius = connectivity_radius
        self.register_buffer('vel_mean', torch.tensor(vel_stats['mean']))
        self.register_buffer('vel_std', torch.tensor(vel_stats['std']))
        self.box_boundaries = box_boundaries
        self.self_edges = self_edges

        # Add the average amount of noise to the standard deviation
        # in the velocity statistics.
        self.vel_std = (self.vel_std.square() + (cum_vel_noise_std**2)/2).sqrt()

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

        # Combine all the tensors into a single masked batched tensor, padding
        # with dummy particles as necessary.
        device = dps[0].materials.device
        n_datapoints = len(dps)
        max_n_particles = max([dp.materials.shape[0] for dp in dps])

        node_mask = torch.zeros(n_datapoints, max_n_particles,
                                dtype=torch.bool, device=device)
        mat = torch.zeros(n_datapoints, max_n_particles,
                          dtype=torch.int64, device=device)
        pos = torch.zeros(n_datapoints, max_n_particles, self.physical_dim,
                          dtype=torch.float32, device=device)
        vel = torch.zeros(n_datapoints, max_n_particles, self.n_previous_velocities, self.physical_dim,
                          dtype=torch.float32, device=device)
        for i, dp in enumerate(dps):
            n_particles = dp.materials.shape[0]
            node_mask[i, :n_particles] = True
            mat[i, :n_particles] = dp.materials
            pos[i, :n_particles] = dp.positions
            vel[i, :n_particles] = dp.velocities

        # Normalize velocities.
        vel = (vel - self.vel_mean) / self.vel_std

        # Make a one-hot representation of the materials, then embed them.
        materials = torch.nn.functional.one_hot(mat, self.n_materials).to(torch.float32)
        materials = self.material_encoder(materials)

        # Concatenate the velocities, materials, and (if present) distance
        # to orthogonal walls, clipped to `connectivity_radius`.
        velocities = vel.view(n_datapoints, max_n_particles, -1)
        node_features = torch.cat([velocities, materials], dim=-1)
        if self.box_boundaries is not None:
            wall_dist = torch.zeros(n_datapoints, max_n_particles, self.physical_dim, 2,
                                    device=device)
            for i in range(self.physical_dim):
                wall_dist[:, :, i, 0] = pos[:, :, i] - self.box_boundaries[i][0]
                wall_dist[:, :, i, 1] = self.box_boundaries[i][1] - pos[:, :, i]
            wall_dist = wall_dist.view(n_datapoints, max_n_particles, -1)
            normalized_wall_dist = torch.clamp(wall_dist/self.connectivity_radius,
                                               min=-1.0, max=1.0)
            node_features = torch.cat([node_features, normalized_wall_dist], dim=-1)
        nodes = self.node_encoder(node_features)

        # Identify neighbors.
        # We do this using brute force on the GPU rather than using a k-d tree
        # on the CPU, as it's about 20x as fast (tested on 2.95k particles).
        left = pos.unsqueeze(2)  # n_datapoints x max_n_particles x 1 x physical_dim
        right = pos.unsqueeze(1) # n_datapoints x 1 x max_n_particles x physical_dim
        edge_mask = (left - right).square().sum(dim=-1) < self.connectivity_radius**2
        edge_mask &= node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
        if not self.self_edges:
            edge_mask.diagonal(dim1=1, dim2=2).fill_(False)
        neighbor_idxs = edge_mask.nonzero()

        # Encode the edges for each pair of neighbors.
        receiver_pos = pos[neighbor_idxs[:, 0], neighbor_idxs[:, 1]]
        sender_pos = pos[neighbor_idxs[:, 0], neighbor_idxs[:, 2]]
        normalized_displacements = (receiver_pos - sender_pos)/self.connectivity_radius
        distances = torch.norm(normalized_displacements, dim=1, keepdim=True)
        edge_features = torch.cat([normalized_displacements, distances], dim=1)
        edges = self.edge_encoder(edge_features)

        return nodes, node_mask, edges, neighbor_idxs

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
        # Precompute the indices for use of scatter_add in each layer.
        scatter_index = neighbor_idxs[:, [0]]*nodes.shape[1] + neighbor_idxs[:, [1]]
        scatter_index = scatter_index.broadcast_to(edges.shape)

        # Iterate over the layers.
        for edge_updater, edge_layer_norm, node_updater, node_layer_norm in zip(
            self.edge_updaters, self.edge_layer_norms, self.node_updaters, self.node_layer_norms):
            # Update the edges.
            receivers = nodes[neighbor_idxs[:, 0], neighbor_idxs[:, 1]]
            senders = nodes[neighbor_idxs[:, 0], neighbor_idxs[:, 2]]
            edges_out = edge_updater(torch.cat([edges, receivers, senders], dim=-1))
            edges_out = edge_layer_norm(edges_out)

            # Aggregate the edges for each node. We do this with node indices flattened
            # across the batch dimension so we can use scatter_add.
            aggregated_edges = torch.zeros(nodes.shape[0]*nodes.shape[1], edges_out.shape[-1],
                                           device=edges_out.device)
            aggregated_edges.scatter_add_(0, scatter_index, edges_out)
            aggregated_edges = aggregated_edges.view(*nodes.shape[:2], -1)

            # Update the nodes.
            nodes_out = node_updater(torch.cat([aggregated_edges, nodes], dim=-1))
            nodes_out = node_layer_norm(nodes_out)

            # Residual connection.
            edges = edges + edges_out
            nodes = nodes + nodes_out

        return nodes, edges, neighbor_idxs

class Decoder(MLP):
    """Decodes graph node output into predicted acceleration."""
    def __init__(self, physical_dim, acc_stats, cum_acc_noise_std, node_embedding_dim=128):
        super().__init__(node_embedding_dim, physical_dim)
        self.register_buffer('acc_mean', torch.tensor(acc_stats['mean']))
        self.register_buffer('acc_std', torch.tensor(acc_stats['std']))

        # Add the average amount of noise to the standard deviation
        # in the acceleration statistics.
        self.acc_std = (self.acc_std.square() + (cum_acc_noise_std**2)/2).sqrt()

    def forward(self, nodes, node_mask):
        acc = super().forward(nodes)
        # De-normalize.
        acc = self.acc_mean + acc*self.acc_std
        return acc*node_mask.unsqueeze(-1)

class GNS(torch.nn.Module):
    """Graph network-based simulator for predicting particle acceleration,
    composed of an Encoder, Processor, and Decoder."""
    def __init__(self, n_materials, physical_dim, n_previous_velocities,
                 connectivity_radius, normalization_stats, cum_vel_noise_std,
                 box_boundaries=None, n_processor_layers=10):
        super().__init__()

        self.encoder = Encoder(n_materials, physical_dim, n_previous_velocities,
                               connectivity_radius, normalization_stats['vel'],
                               cum_vel_noise_std, box_boundaries=box_boundaries)
        self.processor = Processor(n_layers=n_processor_layers)
        self.decoder = Decoder(physical_dim, normalization_stats['acc'], cum_vel_noise_std)

    def forward(self, dp):
        nodes, node_mask, edges, neighbor_idxs = self.encoder(dp)
        nodes, edges, neighbor_idxs = self.processor(nodes, edges, neighbor_idxs)
        accelerations = self.decoder(nodes, node_mask)

        # The modules return data with the batch dimension, so we remove it
        # if that's not how it was given to us.
        if isinstance(dp, TorchDatapoint):
            return accelerations[0]
        else:
            return accelerations

    def get_loss(self, dp, mean=False):
        """Return the total or mean loss across all particles in a batch, along
        with the total number of particles."""
        if isinstance(dp, TorchDatapoint):
            dps = [dp]
        else:
            dps = dp
        acc_hat = self.forward(dps)

        loss = 0.0
        total_n = 0
        for i, dp in enumerate(dps):
            n = dp.target_accelerations.shape[0]
            loss += (acc_hat[i, :n] - dp.target_accelerations).square().sum()
            total_n += n

        if mean:
            loss /= total_n
        return loss, total_n
