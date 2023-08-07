#!/usr/bin/env python3

"""
For analysis of sorting efficacy under different lambda_P values for XEN cells.

A script to run the CPM model, given a set of bootstrap-sampled adhesion matrices.

Runs for a set of lambda_P multipliers, in series. With the same bootstrap-sampled adhesion matrix.

Run from the command-line:

e.g. python run_softstiff.py 72

where 72 defines the bootstrap adhesion matrix that is to be used for parameterising the CPM.

See run_scripts/make_adhesion_matrices.py for details on the bootstraping procedure.
"""
import os
import sys
sys.dont_write_bytecode = True

SCRIPT_DIR = "../"
sys.path.append(os.path.dirname(SCRIPT_DIR))

import time

import numpy as np
import json
from CPM.cpm import CPM
from scipy import sparse
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import center_of_mass
from joblib import delayed, Parallel


def get_adj(I_sparse):
    """
    Calculate the adjacency matrix between each of the cells.

    :param I_sparse: csr-sparse (scipy.sparse) format of cell-indices, i.e. the I matrix.
    :return:
    """
    I = I_sparse.toarray()
    vals = []
    neighs = []
    perim_neighbour_reduced = np.array([[0, 1], [1, 0], [1, 1]])

    for i, j in perim_neighbour_reduced:
        rolled = np.roll(np.roll(I, i, axis=0), j, axis=1)
        mask = I != rolled
        vals += list(I[mask])
        neighs += list(rolled[mask])
    adj = sparse.csr_matrix(
        sparse.coo_matrix(([True] * len(vals), (vals, neighs)), shape=(np.unique(I_sparse.toarray()).size, np.unique(I_sparse.toarray()).size)))
    adj += adj.T
    return adj


def remove_non_attached(I_sparse_,c_types_):
    """
    For a given I matrix (in sparse format, I_sparse_), find the cells that have only 'medium' neighbours and are
    hence non-attached to the main aggregate and remove them.

    Strictly, the cells that remain after this removal are those in the largest connected component in the adjacency
    matrix.
    :param I_sparse_: csr-sparse (scipy.sparse) format of cell-indices, i.e. the I matrix.
    :return: I_sparse, (I matrix, where all non-attached cells are replaced by medium = 0), and c_types_i, the
    cell_type vector for the remaining cells.
    """
    c_types = np.concatenate(((0,),c_types_))
    I_sparse = I_sparse_.copy()
    adj = get_adj(I_sparse)
    n_cc, cc_id = sparse.csgraph.connected_components(adj[1:, 1:])
    c_types_i = c_types.copy()
    if n_cc > 1:
        dom_id = np.argmax(np.bincount(cc_id))
        mask = cc_id != dom_id
        cids_drop = np.nonzero(mask)[0]
        for cid in cids_drop:
            I_sparse[I_sparse == cid + 1] = 0
            c_types_i[cid + 1] = 0
    return I_sparse, c_types_i


def enveloping_score(I_sparse, c_types_i):
    """
    Define whether a cell is enveloping or not.

    A cell is enveloping if its centre of mass lies outside of its perimeter.

    :param I_sparse_: csr-sparse (scipy.sparse) format of cell-indices, i.e. the I matrix.
    :param c_types_i: cell_type index vector, which has been stripped of non-attached cells
    (see **remove_non_attached**).
    :return: Boolean vector, True if enveloping, one per cell.
    """
    C = c_types_i.take(I_sparse.toarray())
    is_enveloping = np.zeros((2), dtype=bool)
    for k, i in enumerate([0,2]):
        j = i + 1
        mid_pt = np.array(center_of_mass(C == j))
        mid_pt_coord = np.round(mid_pt).astype(int)
        mid_pt_type = C[mid_pt_coord[0], mid_pt_coord[1]]
        is_enveloping[k] = mid_pt_type != j
    return is_enveloping

def enveloping_score2(I_sparse, c_types_i):
    """
    Define whether a cell is enveloping or not.

    A cell is enveloping if its centre of mass lies outside of its perimeter.

    :param I_sparse_: csr-sparse (scipy.sparse) format of cell-indices, i.e. the I matrix.
    :param c_types_i: cell_type index vector, which has been stripped of non-attached cells
    (see **remove_non_attached**).
    :return: Boolean vector, True if enveloping, one per cell.
    """
    C = c_types_i.take(I_sparse.toarray())
    mid_pt = np.array(center_of_mass(C !=0))

    is_enveloping = np.zeros((2), dtype=np.float64)
    for k, i in enumerate([1,3]):
        distance_sum = 0
        num_ones = np.sum(C == i)

        for idx in np.argwhere(C == i):
            distance = np.linalg.norm(mid_pt - idx)
            distance_sum += distance

        is_enveloping[k] = distance_sum / num_ones

    return is_enveloping



if __name__ == "__main__":

    # Establish the directory structure for saving.
    if not os.path.exists("../results"):
        os.mkdir("results")

    if not os.path.exists("../results/param_scan"):
        os.mkdir("../results/param_scan")

    if not os.path.exists("../results/param_scan/sims"):
        os.mkdir("../results/param_scan/sims")

    if not os.path.exists("../results/param_scan/analysis"):
        os.mkdir("../results/param_scan/analysis")

    # Get the index of the boostrap adhesion matrix to be used to parameterise the CPM. From the command-line.
    iter_i = int(sys.argv[1])

    # Define the range of lambda_P multiples to be spanned across in the parameter scan.
    lambda_P_mult_range = np.flip(np.linspace(0.2, 1, 10))
    ES_XEN_range = np.linspace(0.38398139838304457,1.9436504565,10)
    XEN_XEN_range = np.linspace(0.38398139838304457,1.9436504565,10)
    seed_range = np.arange(10)

    lP,EX,XX,S = np.meshgrid(lambda_P_mult_range,ES_XEN_range,XEN_XEN_range,seed_range,indexing="ij")
    lP_f,EX_f, XX_f, S_f = lP.ravel(),EX.ravel(),XX.ravel(),S.ravel()

    total_N = lP_f.size
    N_repeat = seed_range
    N_blocks = int(total_N/N_repeat)
    range_to_sample = np.arange(N_repeat*iter_i,N_repeat*(iter_i+1))

    def do_simulation(i,iteration):

        # Define the parameters.
        A0 = 30
        P0 = 0
        lambda_A = 1
        lambda_P = 0.2
        b_e = -0.5

        # Define the W-matrix. Needed, given the architecture, but actually not used in practice, as is replaced by the
        # bootstrapped adhesion matrices.
        W = np.array([[b_e, b_e, b_e, b_e],
                      [b_e, 1.911305, 0.494644, 0.505116],
                      [b_e, 0.494644, 2.161360, 0.420959],
                      [b_e, 0.505116, 0.420959, 0.529589]]) * 6.02



        params = {"A0": [A0, A0, A0],
                  "P0": [P0, P0, P0],
                  "lambda_A": [lambda_A, lambda_A, lambda_A],
                  "lambda_P": [lambda_P, lambda_P, lambda_P * lP_f[i]],
                  "W": W,
                  "T": 15}
        cpm = CPM(params)
        cpm.make_grid(100, 100)
        N_cell_dict = {"E": 12, "T": 0, "X": 12}
        cpm.generate_cells(N_cell_dict=N_cell_dict)
        cpm.make_init("circle", np.sqrt(params["A0"][0] / np.pi) * 0.8, np.sqrt(params["A0"][0] / np.pi) * 0.2)

        # adhesion_dict = json.load(open("raw_data/adhesion_dict.json"))
        # ES_ES = np.array(adhesion_dict["ES-ES"]).mean()
        # ES_XEN = np.array(adhesion_dict["XEN-ES"]).mean()
        # XEN_XEN = np.array(adhesion_dict["XEN-XEN"]).mean()
        ES_ES = 1.9436504565
        ES_XEN = EX_f[i]#0.8329231603358208
        XEN_XEN = XX_f[i]#0.5572779603960396


        adhesion_vals_full = np.zeros((cpm.n_cells+1,cpm.n_cells+1))
        adhesion_vals_full[0] = b_e * cpm.lambda_P
        adhesion_vals_full[:, 0] = b_e * cpm.lambda_P
        adhesion_vals_full[0, 0] = 0
        adhesion_vals_full[1:1+N_cell_dict["E"],1:1+N_cell_dict["E"]] = ES_ES
        adhesion_vals_full[-N_cell_dict["X"]:,-N_cell_dict["X"]:] = XEN_XEN
        adhesion_vals_full[1:1+N_cell_dict["E"],-N_cell_dict["X"]:] = ES_XEN
        adhesion_vals_full[-N_cell_dict["X"]:,1:1+N_cell_dict["E"]] = ES_XEN

        cpm.J = -adhesion_vals_full * 6

        cpm.get_J_diff()
        cpm.I = np.load("../initialisations/%d.npy"%iteration)
        cpm.simulate(int(1e5), int(1000), initialize=False, J0=-8)
        I_save_sparse = [None] * len(cpm.I_save)
        for i, I in enumerate(cpm.I_save):
            I_save_sparse[i] = sparse.csr_matrix(I)
        enveloping_scores = [np.zeros((len(I_save_sparse),2)),np.zeros((len(I_save_sparse),2))]
        for i, I_sparse in enumerate(I_save_sparse):
            I_sparse,c_types_i =  remove_non_attached(I_sparse, cpm.c_types)
            enveloping_scores[0][i] = enveloping_score(I_sparse,c_types_i)
            enveloping_scores[1][i] = enveloping_score2(I_sparse,c_types_i)

        df = pd.DataFrame({"env1_E":enveloping_scores[0][:,0],"env1_X":enveloping_scores[0][:,1],"env2_E":enveloping_scores[1][:,0],"env2_X":enveloping_scores[1][:,1]})
        df.to_csv("../results/param_scan/analysis/%d.csv"%i)
        cpm.save_simulation("../results/param_scan/sims", str(i))


    Parallel(n_jobs=10, backend="loky", prefer="threads")(delayed(do_simulation)(i, j) for i,j in zip(range_to_sample,seed_range))
