import math
import torch
import numpy as np
import scipy as sp
import networkx as nx
import sys
from collections import deque
from functools import partial
from typing import Any, Optional
from torch_geometric.data import Data
from torch_geometric.utils import degree


def bfs_shortest_path(source, G, max_n, cutoff):

    prev_node = [-1] * max_n
    dist = [cutoff] * max_n
    prev_edge_id = [-1] * max_n
    
    queue = deque()
    queue.append(source)
    
    dist[source] = 0
    prev_node[source] = source
    
    while queue:
        v = queue.popleft()
        if dist[v] >= cutoff:
            continue
        for u in G.adj[v]:
            if prev_node[u] < 0:
                prev_node[u] = v
                prev_edge_id[u] = G[u][v]['id']
                dist[u] = dist[v] + 1
                queue.append(u)
                
    return torch.tensor(dist), torch.tensor(prev_node), torch.tensor(prev_edge_id)

def add_reverse_edges(data):
  edge_list = data.edge_index
  undirected_edge_list = torch.cat((edge_list, edge_list.flip(0)), 1)
  data.edge_index = undirected_edge_list
  return data

def add_self_loops(data):
  edge_list = data.edge_index
  vertex_list = torch.unique(edge_list)
  data.edge_index = torch.cat((edge_list, vertex_list.expand(2, -1)), 1)
  if hasattr(data, 'edge_attr') and data.edge_attr is not None:
     assert data.edge_attr.shape[0] == len(edge_list)
     self_loop_feats = torch.zeros(len(vertex_list), data.edge_attr.shape[1])
     data.edge_attr = torch.cat((data.edge_attr, self_loop_feats))

  return data

def add_dist_features(data, max_n, is_undirected, cutoff=None):
    
    if cutoff is None:
        cutoff = np.inf
    
    n = data.num_nodes
    if is_undirected:
        G = nx.Graph()
    else:
        G = nx.DiGraph()
    G.add_nodes_from(range(n))
    
    for i, (u, v) in enumerate(data.edge_index.t().tolist()):
        if is_undirected and u > v:
            continue
        G.add_edge(u, v)
        G[u][v]['id'] = i

    dist_factors = list(map(partial(bfs_shortest_path, G = G, max_n = max_n, cutoff=cutoff), list(range(n))))
    dist = list(map(lambda x: x[0], dist_factors))
    prev_node = list(map(lambda x: x[1], dist_factors))
    prev_edge_id = list(map(lambda x: x[2], dist_factors))

    data.dist = torch.stack(dist)
    data.prev_node = torch.stack(prev_node)
    data.prev_edge_id = torch.stack(prev_edge_id)
    
    data.in_degree = degree(data.edge_index[1])
    data.out_degree = degree(data.edge_index[0])
    
    return data


def find_path(data, source, target):
    if data.prev_nodes[source][target] is None:
        return None
    
    path = []
    node = target
    while node != source:
        path.append(data.prev_edge_id[source][node])
        node = data.prev_nodes[source][node]
        
    path = path[::-1]
    
    return path


def incidence_matrix(senders: np.ndarray, receivers: np.ndarray, n: Optional[int] = None) -> Any:
  """Creates the edge-node incidence matrix for the given edge list.
  The edge list should be symmetric, and there should not be any isolated nodes.
  Args:
    senders: The sender nodes of the graph.
    receivers: The receiver nodes of the graph.
  Returns:
    A sparse incidence matrix
  """

  if n is None:
    n = senders.max() + 1
  m = senders.shape[0]
  rows = list(range(m)) + list(range(m))
  cols = senders.tolist() + receivers.tolist()
  vals = [-1.0] * m + [+1.0] * m
  return sp.sparse.csc_matrix((vals, (rows, cols)), shape=(m, n))


def incidence_matrix_rowcol(senders: np.ndarray, receivers: np.ndarray) -> Any:
  """Returns row list and col list for incidence matrix.
  Args:
    senders: The sender nodes of the graph.
    receivers: The receiver nodes of the graph.
  Returns:
    A sparse incidence matrix
  """
  m = senders.shape[0]
  rows = list(range(m)) + list(range(m))
  cols = senders.tolist() + receivers.tolist()
  return rows, cols


def sqrt_conductance_matrix(senders: np.ndarray, weights: np.ndarray) -> Any:
  """Creates the square root of conductance matrix."""
  m = senders.shape[0]
  rows = list(range(m))
  vals = np.sqrt(weights / 2.0)
  return sp.sparse.csc_matrix((vals, (rows, rows)), shape=(m, m))


def laplacian_matrix(senders: np.ndarray, receivers: np.ndarray,
        weights: Optional[np.ndarray] = None, n: Optional[int] = None) -> Any:
  """Creates the laplacian matrix for given edge list.
  The edge list should be symmetric, and there should not be any isolated nodes.
  Args:
    senders: The sender nodes of the graph
    receivers: The receiver nodes of the graph
    weights: The weights of the edges
  Returns:
    A sparse Laplacian matrix
  """

  if weights is None:
    weights = 0*senders + 1

  if n is None:
    n = senders.max()
    if receivers.max() > n:
      n = receivers.max()
    n += 1

  s = senders.tolist() + list(range(n))
  t = receivers.tolist() + list(range(n))
  w = weights.tolist() + [0.0] * n
  adj = sp.sparse.csc_matrix((w, (s, t)), shape=(n, n))
  lap = adj * -1.0
  lap.setdiag(sp.ravel(adj.sum(axis=0)))
  return lap
  

def laplacian_eigenv(senders: np.ndarray,
                     receivers: np.ndarray,
                     weights: Optional[np.ndarray] = None,
                     k=2,
                     n: Optional[int] = None):
    """Computes the k smallest non-trivial eigenvalue and eigenvectors of the Laplacian matrix corresponding to the given graph.
    Skips all constant vector.
    Args:
        senders: The sender nodes of the graph
        receivers: The receiver nodes of the graph
        weights: The weights of the edges
        k: number of eigenvalue/vector pairs (excluding trivial eigenvector)
        n: # of nodes (optional)
    Returns:
        eigen_values: array of eigenvalues
        eigen_vectors: array of eigenvectors
    """
    m = senders.shape[0]
    if weights is None:
        weights = np.ones(m)

    if n is None:
      n = senders.max()
      if receivers.max() > n:
        n = receivers.max()
      n += 1
    
    lap_mat = laplacian_matrix(senders, receivers, weights, n = n)
    # n = lap_mat.shape[0]
    k = min(n - 2, k + 1)
    # rows of eigenv correspond to graph nodes, cols correspond to eigenvalues
    eigenvals, eigenvecs = sp.sparse.linalg.eigs(lap_mat, k=k, which='SM')
    eigenvals = np.real(eigenvals)
    eigenvecs = np.real(eigenvecs)

    # sort eigenvectors in ascending order of eigenvalues
    sorted_idx = np.argsort(eigenvals)
    eigenvals = eigenvals[sorted_idx]
    eigenvecs = eigenvecs[:, sorted_idx]

    constant_eigenvec_idx = 0

    for i in range(0, k):
        # normalize the i^th eigenvector
        eigenvecs[:, i] = eigenvecs[:, i] / np.sqrt((eigenvecs[:, i]**2).sum())
        if eigenvecs[:, i].var() <= 1e-7:
            constant_eigenvec_idx = i

    non_constant_idx = [*range(0, k)]
    non_constant_idx.remove(constant_eigenvec_idx)

    eigenvals = eigenvals[non_constant_idx]
    eigenvecs = eigenvecs[:, non_constant_idx]

    return eigenvals, eigenvecs


def effective_resistance_embedding(data: Data,
                                   MaxK: int,
                                   weights: Optional[np.ndarray] = None,
                                   accuracy: np.double = 0.1,
                                   which_method: int = 0,
                                   k: int = -1,
                                   n: Optional[int] = None) -> Any:
    """Computes the vector-valued resistive embedding (as opposed to scalar-valued functions along edges provided by the effective_resistances function below) for given graph up to a desired accuracy.
    Args:
    senders: The sender nodes of the graph
    receivers: The receiver nodes of the graph
    weights: The weights of the edges
    accuracy: Target accuracy
    which_method: 0 => choose the most suitable +1 => use random projection
      (approximates effective resistances) -1 => use pseudo-inverse
    Returns:
    Effective resistances embedding (each row corresponds to a node)
    """
    
    senders = data.edge_index[0].cpu().detach().numpy()
    receivers = data.edge_index[1].cpu().detach().numpy()
    
    m = senders.shape[0]
    n = data.num_nodes
    
    if weights is None:
        weights = np.ones(m)

    lap_mat = laplacian_matrix(senders, receivers, weights, n)

    n = lap_mat.shape[0]
    # number of required dimensions is 8 * ln(m)/accuracy^2 if we
    # do random-projection.
    if k == -1:
        k = math.ceil(8 * math.log(m) / (accuracy**2))

    b_mat = incidence_matrix(senders, receivers, n)
    c_sqrt_mat = sqrt_conductance_matrix(receivers, weights)

    # in case of random projection, we need to invert k vectors. if k = Omega(n),
    # it is simply better to (pseudo-)invert the whole laplacian.
    if which_method == -1 or (k >= n / 2 and which_method != +1):
        # embedding = (c_sqrt_mat * b_mat *
        #             sp.linalg.pinv(lap_mat.todense())).transpose()
        try:
          inv_lap_mat = np.linalg.pinv(lap_mat.todense(), hermitian=True).A
          embedding = (c_sqrt_mat * b_mat * inv_lap_mat).transpose()
        except np.linalg.LinAlgError as err:
          print('Could not invert the following matrix: ', lap_mat.todense())
          sys.exit(f"Error {err=}, {type(err)=}")
    else:
        # U C^{1/2} B L^{-1} same as U' L^{-1/2}
        embedding = sp.zeros((n, k))
        for i in range(k):
            y = sp.random.normal(0.0, 1.0 / math.sqrt(k), (1, m))
            y = y * c_sqrt_mat * b_mat
            embedding[..., i], _ = sp.sparse.linalg.cg(lap_mat, y.transpose())

    embedding = torch.tensor(embedding)
    data.er_emb = torch.nn.functional.pad(embedding, (0, MaxK - embedding.shape[1])).float()
    return data


def effective_resistances_from_embedding(
    data: Data,
    normalize_per_node: bool = False) -> Any:
    
    """Computes the effective resistances for given graph using the given embedding.
    Args:
    data should have er_emb feature
    senders: The sender nodes of the graph
    receivers: The receiver nodes of the graph
    normalize_per_node: If true, will normalize the er's so that the sum for
      each node is 1.
    Returns:
    Effective resistances.
    """
    
    senders = data.edge_index[0].cpu().detach().numpy()
    receivers = data.edge_index[1].cpu().detach().numpy()
    embedding = data.er_emb.cpu().detach().numpy()
    
    m = senders.shape[0]
    n = data.num_nodes
    ers = ((incidence_matrix(senders, receivers, n=n) * embedding)**2).sum(axis=1)
    if normalize_per_node:
        sums = sp.zeros((n, 1))
        for _, t, er in zip(senders, receivers, ers):
            sums[t] += er

        for i in range(m):
            ers[i] /= sums[receivers[i]]

    
    ers = torch.Tensor(ers).view(data.edge_index.shape[1],  -1).float()
#     if getattr(data, 'edge_attr') is None:
#         data.edge_attr = ers
#     else:
#         data.edge_attr = torch.cat([data.edge_attr, ers], dim=1)
    data.er_edge = ers
    return data


def effective_resistances(data: Data,
                          weights: Optional[np.ndarray] = None,
                          accuracy: float = 0.1,
                          which_method: int = 0) -> np.ndarray:
    """Computes the effective resistances for given graph up to a desired accuracy.
    Args:
    senders: The sender nodes of the graph
    receivers: The receiver nodes of the graph
    weights: The weights of the edges
    accuracy: Target accuracy
    which_method: 0 => choose the most suitable +1 => use random projection
      (approximates effective resistances) -1 => use pseudo-inverse
    Returns:
    Effective resistances.
    """

    n = data.num_nodes
    senders = data.edge_index[0].cpu().detach().numpy()
    receivers = data.edge_index[1].cpu().detach().numpy()
    m = senders.shape[0]
    
    if weights is None:
        weights = np.ones(m)

    lap_mat = laplacian_matrix(senders, receivers, weights, n)

    # n = lap_mat.shape[0]
    # number of required dimensions is 8 * ln(m)/accuracy^2 if we
    # do random-projection.
    k = math.ceil(8 * math.log(m) / (accuracy**2))

    # in case of random projection, we need to invert k vectors. if k = Omega(n),
    # it is simply better to (pseudo-)invert the whole laplacian.
    if which_method == -1 or (k >= n / 2 and which_method != +1):
        # embedding = sp.linalg.pinv(lap_mat.todense())
        try:
          inv_lap_mat = np.linalg.pinv(lap_mat.todense(), hermitian=True).A
          embedding = (c_sqrt_mat * b_mat * inv_lap_mat).transpose()
        except np.linalg.LinAlgError as err:
          print('Could not invert the following matrix: ', lap_mat.todense())
          sys.exit(f"Error {err=}, {type(err)=}")

        
        def eff_resistance(s, t):
            return embedding[s, s] + embedding[t, t] - embedding[s, t] - embedding[t, s]

        ers = np.array([eff_resistance(s, t) for s, t in zip(senders, receivers)])

    else:
        b_mat = incidence_matrix(senders, receivers, n)
        c_sqrt_mat = sqrt_conductance_matrix(senders, weights)

        # U C^{1/2} B L^{-1} same as U' L^{-1/2}

        embedding = sp.zeros((n, k))
        for i in range(k):
            y = sp.random.normal(0.0, 1.0 / math.sqrt(k), (1, m))
            y = y * c_sqrt_mat * b_mat
            embedding[..., i], _ = sp.sparse.linalg.cg(lap_mat, y.transpose())

        d = b_mat * embedding
        ers = (d**2).sum(axis=1)
        
    ers = torch.Tensor(ers).view(data.edge_index.shape[1], 1).float()

#     if getattr(data, 'edge_attr') is None:
#         data.edge_attr = ers
#     else:
#         data.edge_attr = torch.cat([data.edge_attr, ers], dim=1)

    data.er_edges = ers
        
    return data
