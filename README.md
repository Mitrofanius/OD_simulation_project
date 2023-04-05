# OD Simulation


## Description
Origin-Destination Flow Estimation from link flows (no prior infromation).
An attempt to implement an "Algorithm 1" (so far only the single step model) from  
*Xia, Jingyuan, Wei Dai, John Polak, and Michel Bierlaire. "Dimension Reduction for Origin-Destination Flow Estimation: Blind Estimation Made Possible." arXiv preprint arXiv:1810.06077 (2018)*.

## Project structure
*./utils/network_graph.py* : here you can find a class capable of accepting a graph in form of a python dictionary and generating traffic assignment matrix for o-flows model according to a description in the article mentioned above.

*./notebooks/o_flows_quality_estimation* contains an example of usage of the implemented algorithm on synthetic data and attained results.

*./models/O_flows_linear.py* : implementation of the "Algorithm 1" as far as it was understood.
Constraints C4 and C5 were relaxed because their representation in the framework of quadratic programming would demand ridiculously huge matrices.


## Future work
1) Authors of the article seemingly don't use quadratic programming. It would be great to understand what technique they apply to optimize a matrix with respect to 4 affine constraints.

2) Run our implementation on a real world dataset, e.g. GEANT network data. Unfortunatelly, I can't see how to map the structure of GEANT dataset to the format described in the article (we need an assignment matrix : A_ij,od_ - fraction of traffic from o to d going through link ij; GEANT dataset provides only the flows between od-pairs).

3) Test how severe are constraints C4 and C5 violated in the optimized traffic assignment matrix when they were absent during optimization. Question: can they be relaxed in order to spare memory when training a model with more parameters?

