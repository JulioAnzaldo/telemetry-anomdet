# src/telemetry_anomdet/models/deep/_net.py

"""
Torch network internals for graph based detectors.

This module imports ``torch`` at the top level and will raise ``ImportError``
if torch is not installed. It is imported lazily by the detector wrappers so
that the base ``telemetry_anomdet`` install (which does not depend on torch) can
still import the models package. Install the deep extra to use it::

    uv sync --extra deep

Two pieces live here:

``GATEncoder``
    The shared graph attention spatial encoder. Each feature channel gets a
    learned embedding ``v_i``; a directed graph is built from the top-k
    cosine similarities between embeddings (learned structure, recomputed each
    forward pass); a graph attention layer aggregates each node's neighbours with
    attention conditioned on both the node features and embeddings; a
    (swappable) nonlinearity and an embedding gate produce a per-node
    representation. Follows Deng & Hooi (AAAI 2021) for GDN, and the attention +
    post aggregation activation structure of Wang et al.'s KGL (KAN-GAT). The
    activation is injectable so a KAN variant can replace the default ReLU
    (KGL eq. 3) without touching the graph/attention machinery.

``GDNNet``
    ``GATEncoder`` plus a per-node MLP head that forecasts each sensor's next
    value. The detector wrapper (``gdn.py``) turns per-sensor forecast errors
    into the graph deviation anomaly score.

These networks are intentionally decoupled from the ``BaseDetector`` API so they
can be consumed on their own (e.g. by a symbolic distillation or explainability
pass) without dragging in the detector plumbing.
"""

from __future__ import annotations

import torch
from torch import nn


def topk_graph(embeddings: torch.Tensor, k: int) -> torch.Tensor:
    """
    Build a directed top-k similarity graph over node embeddings.

    For each node i, connect it to the k nodes with the highest cosine
    similarity (excluding itself). Returns a boolean adjacency matrix where
    ``adj[i, j]`` is True when j is a neighbour of i.

    Parameters
    ----------
    embeddings : torch.Tensor, shape (n_nodes, embed_dim)
        Learned per-sensor embedding vectors.
    k : int
        Number of neighbours to retain per node. Clamped to n_nodes - 1.

    Returns
    -------
    adj : torch.Tensor of bool, shape (n_nodes, n_nodes)
    """
    n_nodes = embeddings.shape[0]
    k = max(1, min(k, n_nodes - 1))

    normed = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    sim = normed @ normed.t()  # (n_nodes, n_nodes) cosine similarity
    sim.fill_diagonal_(float("-inf"))  # never select self

    topk_idx = sim.topk(k, dim=1).indices  # (n_nodes, k)
    adj = torch.zeros(n_nodes, n_nodes, dtype=torch.bool, device=embeddings.device)
    adj.scatter_(1, topk_idx, True)
    return adj


class GATEncoder(nn.Module):
    """
    Graph attention spatial encoder shared by the deep detectors.

    Learns per-sensor embeddings, builds a top-k similarity graph, computes
    features and embedding conditioned attention over neighbours, aggregates,
    applies a (swappable) nonlinearity, and gates the result by the node
    embedding. The output is a per-node representation ready for a forecasting
    head.

    Parameters
    ----------
    n_nodes : int
        Number of sensors / feature channels.
    window : int
        Length of the input context per node (window_size - 1 timesteps used
        to forecast the final timestep).
    embed_dim : int, default=64
        Dimensionality of the learned sensor embeddings and hidden features.
        The node feature transform maps ``window -> embed_dim`` so that the
        embedding can gate the aggregated representation element-wise.
    topk : int, default=15
        Number of graph neighbours per node.
    activation : nn.Module or None, default=None
        Nonlinearity applied to the aggregated neighbour features (KGL eq. 3).
        Defaults to ``nn.ReLU()``. A KAN-based variant injects a KAN layer here
        without changing the graph/attention machinery.
    feat : nn.Module or None, default=None
        Node feature transform mapping ``window -> embed_dim``. Defaults to
        ``nn.Linear(window, embed_dim)``. Injectable for the same reason as
        ``activation``: the AR-KAN variants put an ``ARKANFeatures`` here so each
        node's own history passes through a nonlinearity before the graph ever
        sees it. Anything accepting ``(batch, n_nodes, window)`` and returning
        ``(batch, n_nodes, embed_dim)`` works; the encoder does not inspect it.

    Notes
    -----
    Forward input is ``x`` of shape ``(batch, n_nodes, window)`` and the output
    is the per-node representation of shape ``(batch, n_nodes, embed_dim)``.
    """

    def __init__(
        self,
        n_nodes: int,
        window: int,
        embed_dim: int = 64,
        topk: int = 15,
        activation: nn.Module | None = None,
        feat: nn.Module | None = None,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.window = window
        self.embed_dim = embed_dim
        self.topk = topk

        self.embedding = nn.Embedding(n_nodes, embed_dim)
        # Node feature transform W: raw window values -> hidden features.
        self.feat = feat if feat is not None else nn.Linear(window, embed_dim)
        # Attention over the concatenation of the source and target node
        # descriptors g_i = [v_i | W x_i] (GAT-style, feature- and
        # embedding-conditioned, per Deng & Hooi eq. 6-8). Input width is
        # 2 * (embed_dim + embed_dim) = 4 * embed_dim.
        self.attn = nn.Linear(4 * embed_dim, 1, bias=False)
        self.leaky = nn.LeakyReLU(0.2)
        # Post aggregation activation (swappable: ReLU by default, KAN in the
        # KAN-GAT variant).
        self.activation = activation if activation is not None else nn.ReLU()

        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode per-sensor windows into graph attention node representations.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, n_nodes, window)

        Returns
        -------
        z : torch.Tensor, shape (batch, n_nodes, embed_dim)
        """
        batch, n_nodes, _ = x.shape

        v = self.embedding.weight  # (n_nodes, embed_dim)
        h = self.feat(x)  # (batch, n_nodes, embed_dim)  == W x_i

        # Learned graph structure from embedding similarity, with self-loops
        # so each node always attends to itself (paper includes A_ii).
        adj = topk_graph(v, self.topk)  # (n_nodes, n_nodes) bool
        adj = adj | torch.eye(n_nodes, dtype=torch.bool, device=x.device)

        # Node descriptor g_i = [v_i | W x_i], broadcast across the batch.
        vb = v.unsqueeze(0).expand(batch, n_nodes, self.embed_dim)
        g = torch.cat([vb, h], dim=-1)  # (batch, n_nodes, 2*embed_dim)

        # Pairwise attention logits: pi(i, j) = LeakyReLU(a^T [g_i | g_j]).
        gi = g.unsqueeze(2).expand(batch, n_nodes, n_nodes, 2 * self.embed_dim)
        gj = g.unsqueeze(1).expand(batch, n_nodes, n_nodes, 2 * self.embed_dim)
        scores = self.leaky(self.attn(torch.cat([gi, gj], dim=-1)).squeeze(-1))

        # Mask non-neighbours to -inf, then softmax over source nodes j.
        mask = adj.unsqueeze(0)  # (1, n_nodes, n_nodes)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=2)  # (batch, n_nodes, n_nodes)

        # Aggregate neighbour features, activate, then gate by the node
        # embedding: z_i = activation(sum_j alpha_ij W x_j) * v_i.
        z = torch.einsum("bij,bjd->bid", weights, h)  # (batch, n_nodes, embed_dim)
        z = self.activation(z)
        z = z * v.unsqueeze(0)  # broadcast (1, n_nodes, embed_dim)
        return z


class GDNNet(nn.Module):
    """
    GDN forecasting network: a ``GATEncoder`` plus a per-node forecast head.

    Parameters
    ----------
    n_nodes : int
        Number of sensors / feature channels.
    window : int
        Length of the input context per node (window_size - 1 timesteps used
        to forecast the final timestep).
    embed_dim : int, default=64
        Dimensionality of the learned sensor embeddings and hidden features.
    topk : int, default=15
        Number of graph neighbours per node.

    Notes
    -----
    Forward input is ``x`` of shape ``(batch, n_nodes, window)`` and the output
    is the one-step forecast of shape ``(batch, n_nodes)``.
    """

    def __init__(self, n_nodes: int, window: int, embed_dim: int = 64, topk: int = 15):
        super().__init__()
        self.n_nodes = n_nodes
        self.window = window
        self.embed_dim = embed_dim
        self.topk = topk

        self.encoder = GATEncoder(n_nodes, window, embed_dim=embed_dim, topk=topk)
        # Per-node output head: gated hidden features -> scalar forecast.
        self.out = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forecast the next value for every sensor.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, n_nodes, window)

        Returns
        -------
        pred : torch.Tensor, shape (batch, n_nodes)
        """
        z = self.encoder(x)  # (batch, n_nodes, embed_dim)
        pred = self.out(z).squeeze(-1)  # (batch, n_nodes)
        return pred
