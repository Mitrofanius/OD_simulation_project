from collections import namedtuple, defaultdict
import numpy as np
import random

# np.random.seed(0)

Edge = namedtuple("Edge", "orig dest cost")
ProbOverElements = namedtuple("ProbOverElements", "elements prob_distr")

class DirectedGraph:
    """
    takes a graph ->
    generates different traffic assignmentMatrices
    """

    def __init__(self, gdict:dict=None, default_cost:int=1):
        self.default_cost = default_cost
        self.gdict, self.nodes, self.edges, self.edge_costs = self._build_graph(gdict)
        self._od_prob_dict = {}
    
    def _build_graph(self, gdict: dict):
        # function seemingly creates some redundant data structures
        # but they may be useful later
        nodes = set()
        edges = set()
        edge_costs = {}
        gdict_without_costs = defaultdict(list)

        for node, neighbours in gdict.items():
            nodes.add(node)
            for neighbour in neighbours:
                if isinstance(neighbour, int):
                    cost = self.default_cost
                    neighbour_node = neighbour
                else:
                    cost = neighbour[1]
                    neighbour_node = neighbour[0]
                
                gdict_without_costs[node].append(neighbour_node)

                if neighbour_node not in gdict:
                    gdict_without_costs[neighbour_node] = []
                    nodes.add(neighbour_node)

                edge = Edge(node, neighbour_node, cost)
                edges.add(edge)
                edge_costs[(edge.orig, edge.dest)] = cost

        if len(gdict_without_costs) < len(gdict):
            gdict_without_costs.update(gdict)
        return gdict_without_costs, nodes, edges, edge_costs
      

    def add_edge(self, orig, dest, cost:int = None):
        if orig not in self.nodes or dest not in self.nodes:
           self.nodes.add(orig)
           self.nodes.add(dest)

        if cost is None:
            cost = self.default_cost

        edge = Edge(orig, dest, cost)
        self.edges.add(edge) 

    def add_vertex(self, node_number):
       self.nodes.add(node_number)

    def get_adj_matrix(self):
        # for integer-denoted nodes only
        if not isinstance(random.choice(list(self.nodes)), int):
            raise ValueError("nodes should be denoted by integers") 
        
        n = len(self.nodes)
        matrix = np.zeros((n, n))
        for edge in self.edges:
            matrix[edge.orig - 1, edge.dest - 1] = edge.cost

        return matrix
    

    def _get_reachable_nodes(self, node_number:int, max_depth:int = 4):
        traversed = self.__get_reachable_nodes_hlp(node_number, max_depth, set())
        
        return traversed
    

    def __get_reachable_nodes_hlp(self, node_number:int, max_depth:int, traversed:set):
        new_nodes = set()
        for edge in self.edges:
            if edge.orig == node_number and edge.dest not in traversed:
                new_nodes.add(edge.dest)
        if new_nodes:
            for node in new_nodes:
                traversed.add(node)
                traversed = self.__get_reachable_nodes_hlp(node, max_depth, traversed)
        
        return traversed


    def _find_paths_of_len(self, od_paths, path:set, max_length:int = 4):
        current_node = path[-1]
        if not self.gdict.get(current_node):
            return
        
        for node in self.gdict.get(current_node):
            if node in path:
                continue

            curr_path = path + [node]

            if len(curr_path) < max_length + 1 and curr_path not in od_paths:
                od_paths.append(curr_path)
                self._find_paths_of_len(od_paths, curr_path, max_length)
            elif len(curr_path) == max_length + 1:
                od_paths.append(curr_path)
        

    def get_od_pairs(self):
        od_paths = self.get_od_paths()
        od_pairs = set()
        for path in od_paths:
            od_pairs.add((path[0], path[-1]))
        return sorted(od_pairs)
    

    def get_od_paths(self, max_length:int = 4):
        od_paths = []
        for node in self.gdict:
            self._find_paths_of_len(od_paths, [node])
        
        return sorted(od_paths)


    def get_o_nodes(self):
        o_nodes = set()        
        for edge in self.edges:
            o_nodes.add(edge.orig)

        return sorted(o_nodes)
    

    def get_distr_o_over_od(self, distr="uniform"):
        if self._od_prob_dict:
            return self._od_prob_dict
        
        od_pairs = self.get_od_pairs()
        o_nodes = self.get_o_nodes()
        
        # 1) disribution over od pairs
        #    which defines the fraction of O-flow x_o for the involved OD flows s_od
        # od_prob_dict = {}
        for node in o_nodes:
            od_for_o = []
            for orig_node, dest_node in od_pairs:
                if node == orig_node:
                    od_for_o.append((node, dest_node))
            
            num_od_pairs = len(od_for_o)

            # np.random.seed(0)
            if distr == "uniform":
                prob_distr = np.random.uniform(0, 1, num_od_pairs) # just random numbers
            else:
                # here try another distribution, e.g. poisson
                pass
            prob_distr /= prob_distr.sum() # normalize -> now a distribution

            self._od_prob_dict[node] = ProbOverElements(od_for_o, prob_distr)
        
        return  self._od_prob_dict
    

    def get_s_od_probs(self):
        # sum of probs paths leading from o equals 1
        od_pairs = []
        probs = []
        distr = self.get_distr_o_over_od()
        for _, od_pairs_probs in distr.items():
            od_pairs += od_pairs_probs.elements
            probs += list(od_pairs_probs.prob_distr)
        
        od_prob_dict = dict(sorted(zip(od_pairs, probs)))

        return od_prob_dict
    

    def get_s_path_probs(self):
        s_paths = []
        probs = []
        distr = self.get_distr_od_over_paths()
        for _, path_probs in distr.items():
            s_paths += path_probs.elements
            probs += list(path_probs.prob_distr)
        
        path_prob_dict = dict(sorted(zip(map(self.path_to_str, s_paths), probs)))

        return path_prob_dict

    
    def get_distr_od_over_paths(self, distr="inv_sq"):
        # 2) For each given OD pair od ∈ L_OD: distribution over paths

        od_probs_dict = self.get_s_od_probs()

        od_pairs = self.get_od_pairs()
        od_paths = self.get_od_paths()
        path_costs = self.get_path_costs()

        pair_paths_probs = []

        od_path_prob_dict = {}

        for pair in od_pairs:
            pair_paths = []
            pair_paths_costs = []
            for path in od_paths:
                if pair[0] == path[0] and pair[1] == path[-1]:
                    pair_paths.append(path)
                    pair_paths_costs.append(path_costs[self.path_to_str(path)])

            # here probabilities are inversely proportional to squares of costs
            # some other distribution may be used as well
            pair_paths_costs = np.array(pair_paths_costs)

            if distr == "inv_sq":
                pair_paths_batch_probs = (1 / pair_paths_costs) ** 2
            elif distr == "uniform":
                pair_paths_batch_probs = np.random.uniform(0, 1, len(pair_paths_costs))
            pair_paths_batch_probs /= pair_paths_batch_probs.sum()

            # this step can also be done later
            # now its a "conditional distribution" p(s_path_in_od) * p(s_od)
            pair_paths_batch_probs *= od_probs_dict[pair]

            od_path_prob_dict[pair] = ProbOverElements(pair_paths, pair_paths_batch_probs)

        return od_path_prob_dict
    

    def create_incidence_matrix(self):
        paths = list(map(self.path_to_str, sorted(self.get_od_paths())))
        links = list(map(self.path_to_str, sorted(self.edge_costs.keys())))
        A_rows = len(links)
        A_columns = len(paths)
        A = np.zeros((A_rows, A_columns))
        for ij in range(A_rows):
            for od in range(A_columns):
                if links[ij] in paths[od]:
                    A[ij, od] = 1

        return A


    def generate_od_assignment_matrix(self):
        paths = list(map(self.path_to_str, sorted(self.get_od_paths())))
        links = list(map(self.path_to_str, sorted(self.edge_costs.keys())))
        od_pairs = list(map(self.path_to_str, sorted(self.get_od_pairs())))

        path_probs = [p_prob[1] for p_prob in sorted(self.get_s_path_probs().items())]
        A_inc_mat = self.create_incidence_matrix()
        A_rows = len(links)
        A_columns = len(od_pairs)
        A = np.zeros((A_rows, A_columns))

        for ij in range(A_rows):
            for od in range(A_columns):
                s_od = 0
                sum_a_ij_s_p = 0
                for p in range(len(paths)):
                    if od_pairs[od][0] == paths[p][0] and od_pairs[od][1] == paths[p][-1]:
                        s_p = path_probs[p]
                        sum_a_ij_s_p += A_inc_mat[ij][p] * s_p
                        s_od += s_p

                A[ij, od] = sum_a_ij_s_p / s_od

        return A
    

    def generate_o_assignment_matrix(self):
        A = self.generate_od_assignment_matrix()
        # od_pairs = list(map(self.path_to_str, self.get_od_pairs()))
        od_pairs = sorted(self.get_od_pairs())
        distr_o_od = self.get_distr_o_over_od()
        s_od_probs = self.get_s_od_probs()

        o_nodes = sorted(self.get_o_nodes())
        rows, cols = A.shape[0], len(o_nodes)

        P = np.zeros((rows, len(o_nodes)))
        for ij in range(rows):
            for o in range(cols):
                sum_a_s = 0
                x_o = 0
                for od, pair in enumerate(od_pairs):
                    if o_nodes[o] == pair[0]:
                        sum_a_s += A[ij, od] * s_od_probs[pair]
                        x_o += s_od_probs[pair]
                
                P[ij, o] = sum_a_s / x_o

        return P

        
    def get_path_costs(self):
        od_paths = self.get_od_paths()

        path_costs = {}
        for path in od_paths:
            cost = 0
            for i in range(len(path) - 1):
                cost += self.edge_costs[(path[i], path[i+1])]
                path_costs[self.path_to_str(path)] = cost
                # path_costs[frozenset(path)] = cost

        return path_costs
    
    def generate_o_flows(self, rng:int = 1000):
        o_nodes = sorted(self.get_o_nodes())
        # o_flows = np.round(np.random.uniform(0, 1, len(o_nodes)) * 1000)
        o_flows = np.random.randint(rng, size=len(o_nodes))

        return o_flows, o_nodes
    
    
    def get_od_from_o_flows(self, o_flows, P_o):
        od_pairs = sorted(self.get_od_pairs())
        o_nodes = sorted(self.get_o_nodes())
        links = list(map(self.path_to_str, sorted(self.edge_costs.keys())))

        od_pairs_inv_dict = dict(zip(od_pairs, [i for i, _ in enumerate(od_pairs)]))

        od_flows = np.zeros(len(od_pairs))

        ij_rows, o_cols = P_o.shape

        for o in range(o_cols):
            for ij in range(ij_rows):
                i = int(links[ij][0])
                d = int(links[ij][1])

                if od_pairs_inv_dict.get((o + 1, d)) is None:
                    continue
                ind_of_od = od_pairs_inv_dict[(o + 1, d)]

                od_flows[ind_of_od] += o_flows[o] * P_o[ij, o]

                d = int(links[ij][0])
                j = int(links[ij][1])
                if od_pairs_inv_dict.get((o + 1, d)) is None:
                    continue
                ind_of_od = od_pairs_inv_dict[(o + 1, d)]
                od_flows[ind_of_od] -= o_flows[o] * P_o[ij, o]

        return od_flows, od_pairs
    
    
    def get_links(self):
        return list(map(self.path_to_str, sorted(self.edge_costs.keys())))

    def path_to_str(self, path:list):
        return "".join(map(str, path))

    

   
graph_dict_5_nodes = { 
   1 : [(2, 1), (3, 1)],
   2 : [1, 4],
   3 : [1, 4],
   4 : [5],
   5 : [4]
}

graph_dict_4_nodes = { 
   1 : [(2, 1), (3, 1)],
   2 : [4],
   3 : [4]
}

graph_dict_3x3 = { 
   1 : [4, 2],
   2 : [5, 3],
   3 : [6],
   4 : [5, 7],
   5 : [8, 6],
   6 : [9],
   7 : [8],
   8 : [9]
}

graph_dict_3x3_2 = { 
   1 : [4, 2],
   2 : [1, 5, 3],
   3 : [2, 6],
   4 : [1, 5, 7],
   5 : [2, 4, 8, 6],
   6 : [3, 5, 9],
   7 : [4, 8],
   8 : [5, 7, 9],
   9 : [8, 6]
}

if __name__ == "__main__":
    # G = DirectedGraph(graph_dict_3x3)
    G = DirectedGraph(graph_dict_4_nodes)
    # print(G.edges)

    # print(G.get_od_pairs())
    # print(len(G.get_od_pairs()))

    # print("od_paths", G.get_od_paths())


    # generate assignment matrix:
    # 1) disribution over od pairs
    #    which defines the fraction of O-flow x_o for the involved OD flows s_od
    # 2) For each given OD pair od ∈ L_OD: distribution over paths

    # print(G.get_distr_o_over_od())
    # print(G.get_path_costs())
    # print(G.path_to_str([1, 2, 3, 4, 5]))
    # print("prob over paths: ", G.get_distr_od_over_paths())

    # print("num of edges: ", len(G.edges))
    # print(G.get_s_od_probs())
    # print((G.get_s_path_probs()))
    # print(sorted(G.get_s_path_probs().items()))
    # print(sorted(G.get_od_paths()))

    # print(G.create_incidence_matrix())

    print(G.generate_od_assignment_matrix())
    print(G.generate_o_assignment_matrix())

    # print(G.create_incidence_matrix())
    # print(G.get_distr_od_over_paths())

    # print(G.get_od_pairs())
    # print(G.get_o_nodes())


