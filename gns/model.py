import torch

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

class Encoder(torch.nn.Module):
    """Encodes a datapoint into the appropriate graph representation, excluding acceleration
    (which is the prediction target)."""
    def __init__(self, n_materials, physical_dim, n_previous_velocities,
                 material_embedding_dim=16, node_embedding_dim=128, edge_embedding_dim=128):
        super().__init__()
        self.n_materials = n_materials
        self.physical_dim = physical_dim
        self.n_previous_velocities = n_previous_velocities

        # Materials are embedded from a one-hot representation.
        self.material_encoder = torch.nn.Linear(n_materials, material_embedding_dim)

        # Nodes are encoded from a concatenation of the previous velocities and
        # material of the corresponding particle.
        self.node_encoder = MLP(n_previous_velocities*physical_dim + material_embedding_dim,
                                 node_embedding_dim)
        self.node_layer_norm = torch.nn.LayerNorm(node_embedding_dim)
        
        # Edges are encoded from a concatenation of the relative position and distance
        # between the two neighboring particles.
        self.edge_encoder = MLP(physical_dim + 1, edge_embedding_dim)
        self.edge_layer_norm = torch.nn.LayerNorm(edge_embedding_dim)
    
    def forward(self, dp):
        # Make a one-hot representation of the materials, then embed them.
        materials = torch.nn.functional.one_hot(dp.materials, self.n_materials).to(torch.float32)
        materials = self.material_encoder(materials)

        # Concatenate the positions, velocities, and materials.
        velocities = dp.velocities.reshape(-1, self.n_previous_velocities*self.physical_dim)
        node_features = torch.cat([velocities, materials], dim=1)
        nodes = self.node_encoder(node_features)
        nodes = self.node_layer_norm(nodes)

        # Compute the relative positions and distances.
        relative_positions = dp.positions[dp.neighbor_idxs[:, 0]] - dp.positions[dp.neighbor_idxs[:, 1]]
        distances = torch.norm(relative_positions, dim=1, keepdim=True)
        edge_features = torch.cat([relative_positions, distances], dim=1)
        edges = self.edge_encoder(edge_features)
        edges = self.edge_layer_norm(edges)

        return nodes, edges, dp.neighbor_idxs

class Processor(torch.nn.Module):
    """`n_layers` graph network blocks, as described in Battaglia et al (2018).
    Relational inductive biases, deep learning, and graph networks.

    Reduction is by summation, and there is no global state."""
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
            receivers = nodes[neighbor_idxs[:, 0]]
            senders = nodes[neighbor_idxs[:, 1]]
            edges = edge_updater(torch.cat([edges, receivers, senders], dim=1))
            edges = edge_layer_norm(edges)

            # Aggregate the edges for each node.
            aggregated_edges = torch.zeros(nodes.shape[0], edges.shape[1])
            index = neighbor_idxs[:, [0]].broadcast_to((edges.shape[0], edges.shape[1]))
            aggregated_edges = aggregated_edges.scatter_add(0, index, edges)

            # Update the nodes.
            nodes = node_updater(torch.cat([aggregated_edges, nodes], dim=1))
            nodes = node_layer_norm(nodes)

        return nodes, edges, neighbor_idxs

class Decoder(MLP):
    """Decodes graph node output into predicted acceleration."""
    def __init__(self, physical_dim, node_embedding_dim=128):
        super().__init__(node_embedding_dim, physical_dim)
