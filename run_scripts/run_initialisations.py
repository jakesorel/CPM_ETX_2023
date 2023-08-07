#!/usr/bin/env python3

import os
import sys
import time

import numpy as np
import json
from CPM.cpm import CPM
import matplotlib.pyplot as plt

if __name__ == "__main__":


    # Establish the directory structure for saving.
    if not os.path.exists("../initialisations"):
        os.mkdir("../initialisations")

    for iteration in range(10):
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
                  "lambda_P": [lambda_P, lambda_P, lambda_P * lpm],
                  "W": W,
                  "T": 15}
        cpm = CPM(params)
        cpm.make_grid(100, 100)
        N_cell_dict = {"E": 12, "T": 0, "X": 12}
        cpm.generate_cells(N_cell_dict=N_cell_dict)
        cpm.make_init("circle", np.sqrt(params["A0"][0] / np.pi) * 0.8, np.sqrt(params["A0"][0] / np.pi) * 0.2)

        cpm.get_J_diff()
        t0 = time.time()
        cpm.initialize(J0=-8,n_initialise_steps=10000)
        np.save("../initialisations/%d"%iteration,cpm.I)
