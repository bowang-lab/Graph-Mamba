import math
import numpy as np
import scipy as sp
from typing import Any, Optional
import torch
from graphgps.transform.dist_transforms import laplacian_eigenv


def generate_random_regular_graph1(num_nodes, degree, rng=None):
  """Generates a random 2d-regular graph with n nodes using permutations algorithm.
  Returns the list of edges. This list is symmetric; i.e., if
  (x, y) is an edge so is (y,x).
  Args:
    num_nodes: Number of nodes in the desired graph.
    degree: Desired degree.
    rng: random number generator
  Returns:
    senders: tail of each edge.
    receivers: head of each edge.
  """

  if rng is None:
    rng = np.random.default_rng()

  senders = [*range(0, num_nodes)] * degree
  receivers = []
  for _ in range(degree):
    receivers.extend(rng.permutation(list(range(num_nodes))).tolist())

  senders, receivers = [*senders, *receivers], [*receivers, *senders]

  senders = np.array(senders)
  receivers = np.array(receivers)

  return senders, receivers



def generate_random_regular_graph2(num_nodes, degree, rng=None):
  """Generates a random 2d-regular graph with n nodes using simple variant of permutations algorithm.
  Returns the list of edges. This list is symmetric; i.e., if
  (x, y) is an edge so is (y,x).
  Args:
    num_nodes: Number of nodes in the desired graph.
    degree: Desired degree.
    rng: random number generator
  Returns:
    senders: tail of each edge.
    receivers: head of each edge.
  """

  if rng is None:
    rng = np.random.default_rng()

  senders = [*range(0, num_nodes)] * degree
  receivers = rng.permutation(senders).tolist()

  senders, receivers = [*senders, *receivers], [*receivers, *senders]

  return senders, receivers


def generate_random_graph_with_hamiltonian_cycles(num_nodes, degree, rng=None):
  """Generates a 2d-regular graph with n nodes using d random hamiltonian cycles.
  Returns the list of edges. This list is symmetric; i.e., if
  (x, y) is an edge so is (y,x).
  Args:
    num_nodes: Number of nodes in the desired graph.
    degree: Desired degree.
    rng: random number generator
  Returns:
    senders: tail of each edge.
    receivers: head of each edge.
  """

  if rng is None:
    rng = np.random.default_rng()

  senders = []
  receivers = []
  for _ in range(degree):
    permutation = rng.permutation(list(range(num_nodes))).tolist()
    for idx, v in enumerate(permutation):
      u = permutation[idx - 1]
      senders.extend([v, u])
      receivers.extend([u, v])

  senders = np.array(senders)
  receivers = np.array(receivers)

  return senders, receivers


def generate_random_expander(data, degree, algorithm, rng=None, max_num_iters=100, exp_index=0):
  """Generates a random d-regular expander graph with n nodes.
  Returns the list of edges. This list is symmetric; i.e., if
  (x, y) is an edge so is (y,x).
  Args:
    num_nodes: Number of nodes in the desired graph.
    degree: Desired degree.
    rng: random number generator
    max_num_iters: maximum number of iterations
  Returns:
    senders: tail of each edge.
    receivers: head of each edge.
  """

  num_nodes = data.num_nodes

  if rng is None:
    rng = np.random.default_rng()
  
  eig_val = -1
  eig_val_lower_bound = max(0, 2 * degree - 2 * math.sqrt(2 * degree - 1) - 0.1)

  max_eig_val_so_far = -1
  max_senders = []
  max_receivers = []
  cur_iter = 1

  if num_nodes <= degree:
    degree = num_nodes - 1    
    
  # if there are too few nodes, random graph generation will fail. in this case, we will
  # add the whole graph.
  if num_nodes <= 10:
    for i in range(num_nodes):
      for j in range(num_nodes):      
        if i != j:
          max_senders.append(i)
          max_receivers.append(j)
  else:
    while eig_val < eig_val_lower_bound and cur_iter <= max_num_iters:
      if algorithm == 'Random-d':
        senders, receivers = generate_random_regular_graph1(num_nodes, degree, rng)
      elif algorithm == 'Random-d-2':
        senders, receivers = generate_random_regular_graph2(num_nodes, degree, rng)
      elif algorithm == 'Hamiltonian':
        senders, receivers = generate_random_graph_with_hamiltonian_cycles(num_nodes, degree, rng)
      else:
        raise ValueError('prep.exp_algorithm should be one of the Random-d or Hamiltonian')
      [eig_val, _] = laplacian_eigenv(senders, receivers, k=1, n=num_nodes)
      if len(eig_val) == 0:
        print("num_nodes = %d, degree = %d, cur_iter = %d, mmax_iters = %d, senders = %d, receivers = %d" %(num_nodes, degree, cur_iter, max_num_iters, len(senders), len(receivers)))
        eig_val = 0
      else:
        eig_val = eig_val[0]

      if eig_val > max_eig_val_so_far:
        max_eig_val_so_far = eig_val
        max_senders = senders
        max_receivers = receivers

      cur_iter += 1

  # eliminate self loops.
  non_loops = [
      *filter(lambda i: max_senders[i] != max_receivers[i], range(0, len(max_senders)))
  ]

  senders = np.array(max_senders)[non_loops]
  receivers = np.array(max_receivers)[non_loops]

  max_senders = torch.tensor(max_senders, dtype=torch.long).view(-1, 1)
  max_receivers = torch.tensor(max_receivers, dtype=torch.long).view(-1, 1)

  if exp_index == 0:
    data.expander_edges = torch.cat([max_senders, max_receivers], dim=1)
  else:
    attrname = f"expander_edges{exp_index}"
    setattr(data, attrname, torch.cat([max_senders, max_receivers], dim=1)) 

  return data
