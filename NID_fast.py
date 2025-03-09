import bisect
import operator
import numpy as np
import torch
from torch.utils import data
#from utils import *
from statistics import mean
import itertools
from math import log
from tqdm import tqdm



def get_nlargest(generator, N, factor=1, minimum=-float("Inf")):

  biglist = []
  for inter in generator:
    snps, strength = inter
    if strength > minimum:
      biglist.append(inter)
    if len(biglist) > factor*N:
      biglist.sort(key=lambda x: x[1], reverse=True)
      del biglist[N:]
      minimum = biglist[-1][1]

  biglist.sort(key=lambda x: x[1], reverse=True)
  del biglist[N:]

  return biglist


def get_nsmallest(generator, N, factor=1, maximum=float("Inf")):

  small_list = []
  for inter in generator:
    snps, strength = inter
    if strength < maximum:
      small_list.append(inter)
    if len(small_list) > factor*N:
      small_list.sort(key=lambda x: x[1], reverse=False)
      del small_list[N:]
      maximum = small_list[-1][1]

  small_list.sort(key=lambda x: x[1], reverse=False)
  del small_list[N:]

  return small_list


def get_weights(model):
    weights = []
    for name, param in model.named_parameters():
        if "neural_network" in name and "weight" in name and len(param.shape) != 1:
            weights.append(param.cpu().detach().numpy())
    return weights

def preprocess_weights(weights):
    w_later = np.abs(weights[-1])
    w_input = np.abs(weights[0])

    for i in range(len(weights) - 2, 0, -1):
        w_later = np.matmul(w_later, np.abs(weights[i]))

    return w_input, w_later


def make_one_indexed(interaction_ranking):
    return [(tuple(np.array(i) + 1), s) for i, s in interaction_ranking]


def interpret_interactions(w_input, w_later, get_main_effects=False):

    interaction_strengths = {}
    for i in range(w_later.shape[1]): # Looping through hidden units
        sorted_hweights = sorted(
            enumerate(w_input[i]), key=lambda x: x[1], reverse=True
        ) # Sorting input weights. Because weights are sorted to begin with, it is possible to not go through all possible 2-500-order interactions
        interaction_candidate = []
        candidate_weights = []
        for j in range(w_input.shape[1]): # Looping through input features
            bisect.insort(interaction_candidate, sorted_hweights[j][0]) # The ID of the input feature in a sorted manner
            # We keep appending input ID's to the list interaction_candidate
            candidate_weights.append(sorted_hweights[j][1]) # The weight of the input feature

            if not get_main_effects and len(interaction_candidate) == 1: # If get_main_effects == False and only 1 ID, the loop is done.
                continue

            interaction_tup = tuple(interaction_candidate)
            if interaction_tup not in interaction_strengths: # Assign strength to zero if the input feature ID's are not in the keys of interaction_strengths
                interaction_strengths[interaction_tup] = 0

            interaction_strength = ( min(candidate_weights) ) * ( np.sum(w_later[:, i]) ) # Computing interaction strength according to paper
            interaction_strengths[interaction_tup] += interaction_strength # Assigning interaction strength to corresponding input ID's and summing over hidden units

    interaction_ranking = sorted(
        interaction_strengths.items(), key=operator.itemgetter(1), reverse=True
    )

    return interaction_ranking


def interpret_pairwise_interactions(w_input, w_later):
    
    p = w_input.shape[1]
    for i in tqdm(range(p)):
        for j in range(p):
            if i < j:
                yield [(i+1, j+1), (np.minimum( w_input[:, i], w_input[:, j] ) * w_later).sum()]
  

def garsons_algorithm(w_input, w_later):
    
    p = w_input.shape[1]
    for i in tqdm(range(p)):
        yield [i+1, (w_input[:,i] * w_later).sum()]


def get_interactions(weights, pairwise=False, one_indexed=False):

    w_input, w_later = preprocess_weights(weights)

    if pairwise:
        interaction_ranking = interpret_pairwise_interactions(w_input, w_later)

    else:
        interaction_ranking = interpret_interactions(w_input, w_later)
        interaction_ranking = prune_redundant_interactions(interaction_ranking)

    return sorted(interaction_ranking, key=lambda x: x[1], reverse=True)


def prune_redundant_interactions(interaction_ranking, max_interactions=500):
    interaction_ranking_pruned = []
    current_superset_inters = []
    for inter, strength in interaction_ranking:
        set_inter = set(inter)
        if len(interaction_ranking_pruned) >= max_interactions:
            break
        subset_inter_skip = False
        update_superset_inters = []
        for superset_inter in current_superset_inters:
            if set_inter < superset_inter:
                subset_inter_skip = True
                break
            elif not (set_inter > superset_inter):
                update_superset_inters.append(superset_inter)
        if subset_inter_skip:
            continue
        current_superset_inters = update_superset_inters
        current_superset_inters.append(set_inter)
        interaction_ranking_pruned.append((inter, strength))

    return interaction_ranking_pruned



def detection_rates(interactions, ground_truth):

    """ Computes the liberal and conservative detection rate.
    The liberal detection rate includes subsets of the ground truth,
    whereas the conservative do not. Allows for results to be printed
    to screen or saved as variable. """

    found_or_subset = np.zeros(ncombs(ground_truth), dtype=bool)
    found = np.zeros(len(ground_truth), dtype=bool)
    found_lib = np.zeros(len(ground_truth), dtype=bool)
    for i in range(0, len(ground_truth)):
        for rank, int in enumerate(interactions):
            snps, strength = int
            if ground_truth[i] == set(snps):
                found[i] = True
                found_lib[i] = True
                found_or_subset[i] = True
            if ground_truth[i] > set(snps): # is ground truth as superset of snps?
                found_or_subset[i] = True
                found_lib[i] = True


    liberal_detection_rate = found_lib.mean()
    conservative_detection_rate = found.mean()
    subset_detection_rate = found_or_subset.mean()

    return liberal_detection_rate, conservative_detection_rate, subset_detection_rate



def conservative_rate(interactions, ground_truth):
    found = np.zeros(len(ground_truth), dtype=bool)
    for i in range(0, len(ground_truth)):
        for rank, inter in enumerate(interactions):
            snps, strength = inter 
            if ground_truth[i] == set(snps):
                found[i] = True

    return found

def liberal_rate(interactions, ground_truth):
    found = np.zeros(len(ground_truth), dtype=bool)
    for i in range(0, len(ground_truth)):
        for rank, inter in enumerate(interactions):
            snps, strength = inter 
            if ground_truth[i] == set(snps):
                found[i] = True
            if ground_truth[i] > set(snps): # is ground truth as superset of snps?
                found[i] = True

    return found


def get_gtrank(interactions, ground_truth):

    ranks = np.empty(len(ground_truth))
    ranks[:] = np.nan
    for i in range(0, len(ground_truth)):
        for rank, inter in enumerate(interactions):
          snps, strength = inter 
          if ground_truth[i] == set(snps):
            ranks[i] = rank

    return ranks


def detect_interactions(
    Xd,
    Yd,
    arch=[256, 128, 64],
    batch_size=100,
    device=torch.device("cpu"),
    seed=None,
    **kwargs
):

    if seed is not None:
        set_seed(seed)

    data_loaders = convert_to_torch_loaders(Xd, Yd, batch_size)

    model = create_mlp([feats.shape[1]] + arch + [1]).to(device)

    model, mlp_loss = train(model, data_loaders, device=device, **kwargs)
    inters = get_interactions(get_weights(model))

    return inters, mlp_loss
