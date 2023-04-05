import os
import sys
import numpy as np

sys.path.insert(1, os.path.dirname(os.path.dirname(__file__)))

from utils.network_graph import DirectedGraph
from utils.opt_functions import quadprog_solve_qp


class OFlowsLinear:
    def __init__(self, graph) -> None:
        self.graph = graph

        self.o_nodes = graph.get_o_nodes()
        self.links = graph.get_links()
        self.num_o_nodes = len(self.o_nodes)
        self.num_links = len(self.links)

        self.C1_mat = np.vstack([-np.eye(self.num_o_nodes) for i in range(self.num_o_nodes)])
        self.C1_vect = np.zeros(self.num_o_nodes ** 2)

        # num of rows and columns of assignment matrix
        n_rows, n_cols = self.num_links, self.num_o_nodes

        self.C2_mat = np.vstack([np.eye(n_rows*n_cols), -np.eye(n_rows * n_cols)]) 
        self.C2_vect = np.hstack([np.ones(n_rows*n_cols), np.zeros(n_rows * n_cols)])
    
        self.C3_mat = np.zeros((n_cols, n_rows * n_cols))
        for j in range(n_cols):
            for i in range(n_rows):
                    if (int(self.links[i][0])) == j + 1:
                        self.C3_mat[j, i * n_cols + j] = 1
        self.C3_vect = np.ones(n_cols)


    def fit(
        self, 
        n_t,
        link_flows_mat,
        num_inits = 10,
        threshold = 1e-5,
        max_iters = 5000,
        verbose=False
        ):

        opt_P_inits = np.zeros((num_inits, self.num_links, self.num_o_nodes))

        for i in range(num_inits):
            graph_h = DirectedGraph(self.graph.get_original_graph_repr())
            opt_P = graph_h.generate_o_assignment_matrix()
            opt_x = np.zeros((self.num_o_nodes, n_t))

            iters = 0
            y_pred = np.zeros(link_flows_mat.shape)

            nmse = (np.linalg.norm(link_flows_mat - y_pred) / np.linalg.norm(link_flows_mat)) **2
            while nmse > threshold and iters < max_iters:
                if verbose and iters >= 1:
                    print(f"NMSE, iteration {iters}: ", nmse)
                opt_x = self._optimize_x(opt_P, n_t, opt_x, link_flows_mat)
                opt_P = self._optimize_P(opt_x, n_t, link_flows_mat)
                y_prev = y_pred
                y_pred = opt_P @ opt_x
                nmse = (np.linalg.norm(link_flows_mat - y_pred) / np.linalg.norm(link_flows_mat)) **2
                iters += 1

            opt_P_inits[i, :, :] = opt_P

        opt_P = (np.sum(opt_P_inits, axis=0) / num_inits)
        opt_x = self._optimize_x(opt_P, n_t, opt_x, link_flows_mat)

        if num_inits != 1:
            iters = 0
            nmse = (np.linalg.norm(link_flows_mat - y_pred) / np.linalg.norm(link_flows_mat)) **2
            while nmse > threshold and iters < max_iters:
                if verbose and iters >= 1:
                    print(f"NMSE, iteration {iters}: ", nmse)
                opt_x = self._optimize_x(opt_P, n_t, opt_x, link_flows_mat)
                opt_P = self._optimize_P(opt_x, n_t, link_flows_mat)
                y_prev = y_pred
                y_pred = opt_P @ opt_x
                nmse = (np.linalg.norm(link_flows_mat - y_pred) / np.linalg.norm(link_flows_mat)) **2
                iters += 1

        return opt_P, opt_x
    
    def _optimize_x(self, opt_P, n_t, opt_x, link_flows_mat):
        M_x = opt_P
        P_x = np.dot(M_x.T, M_x)

        for j in range(n_t):
            q_x = -np.dot(M_x.T, link_flows_mat[:, j])
            opt_x[:, j] = quadprog_solve_qp(P_x, q_x, self.C1_mat, self.C1_vect)
        return opt_x


    def _optimize_P(self, opt_x, n_t, link_flows_mat):
        n_rows, n_cols = self.num_links, self.num_o_nodes


        M = np.zeros((n_rows * n_t, n_rows * n_cols))

        for j in range(n_t):
            for i in range(n_rows):
                M[i + j * n_rows, i*n_cols:i*n_cols+n_cols] = opt_x[:, j]

        P = np.dot(M.T, M)
        b = link_flows_mat.T.reshape(np.prod(link_flows_mat.shape))
        q = -M.T @ b

        G = self.C2_mat
        h = self.C2_vect
        A = self.C3_mat
        b = self.C3_vect

        opt_P = (quadprog_solve_qp(P, q, G, h, A, b))\
                .reshape((self.num_links, self.num_o_nodes))
        
        return opt_P

