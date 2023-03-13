import math
import numpy as np
import scipy as sp
from typing import Any, Optional
import torch
from graphgps.transform.dist_transforms import laplacian_eigenv


def generate_random_regular_graph(num_nodes, degree, rng=None):
  """Generates a random d-regular gåraph with n nodes.
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

  # eliminate self loops.
  non_loops = [
      *filter(lambda i: senders[i] != receivers[i], range(0, len(senders)))
  ]

  senders = np.array(senders)[non_loops]
  receivers = np.array(receivers)[non_loops]

  return senders, receivers


def generate_random_expander(data, degree, rng=None, max_num_iters=100, exp_index=0):
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
  eig_val_lower_bound = max(0, degree - 2 * math.sqrt(degree - 1) - 0.1)

  max_eig_val_so_far = -1
  max_senders = []
  max_receivers = []
  cur_iter = 1

  # (bave): This is a hack.  This should hopefully fix the bug
  if num_nodes <= degree:
    # senders, receivers = generate_random_regular_graph(num_nodes, num_nodes, rng)
    # max_senders = torch.tensor(senders).view(-1, 1)
    # max_receivers = torch.tensor(receivers).view(-1, 1)
    # if exp_index == 0:
    #  data.expander_edges = torch.cat([max_senders, max_receivers], dim=1)
    # else:
    #  attrname = f"expander_edges{exp_index}"
    #  setattr(data, attrname, torch.cat([max_senders, max_receivers], dim=1)) 
    # return data
    degree = num_nodes - 1    
    
  # (ali): if there are too few nodes, random graph generation will fail. in this case, we will
  # add the whole graph.
  if num_nodes <= 10:
    for i in range(num_nodes):
      for j in range(num_nodes):      
        if i != j:
          max_senders.append(i)
          max_receivers.append(j)
  else:
    while eig_val < eig_val_lower_bound and cur_iter <= max_num_iters:
      senders, receivers = generate_random_regular_graph(num_nodes, degree, rng)
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

  max_senders = torch.tensor(max_senders, dtype=torch.long).view(-1, 1)
  max_receivers = torch.tensor(max_receivers, dtype=torch.long).view(-1, 1)
  if exp_index == 0:
    data.expander_edges = torch.cat([max_senders, max_receivers], dim=1)
  else:
    attrname = f"expander_edges{exp_index}"
    setattr(data, attrname, torch.cat([max_senders, max_receivers], dim=1)) 

  return data
