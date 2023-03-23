import numpy
import quadprog


def quadprog_solve_qp(P, q, G, h, A=None, b=None):
    qp_G = .5 * (P + P.T)   # make sure P is symmetric
    qp_a = -q
    if A is not None:
        qp_C = -numpy.vstack([A, G]).T
        qp_b = -numpy.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]



# optimize assignment matrix with known o-flows
links = graph.get_links()

link_flows = P_o @ o_flows
r, c = P_o.shape

M = np.zeros((r, r*c))
for i in range(r):
    M[i, i*c:i*c+c] = o_flows


P = np.dot(M.T, M)
q = np.zeros(r*c)

for i in range(r):
    q[i * c: i * c + c] = link_flows[i] * o_flows

C2_mat = np.vstack([np.eye(r*c), -np.eye(r * c)]) 
C2_vect = np.hstack([np.ones(r*c), np.zeros(r * c)])

C3_mat = np.zeros((c, r * c))
for j in range(c):
      for i in range(r):
            if (int(links[i][0])) == j + 1:
                  C3_mat[j, i * c + j] = 1
C3_vect = np.ones(c)


G = C2_mat
h = C2_vect
A = C3_mat
b = C3_vect

P += np.eye(P.shape[0]) * 1e-8

np.round(quadprog_solve_qp(P, q, G, h, A, b) * 1000)


# C4 I dont understand
# and C5 is hard to code and verify it's correctness
# + it requires a huge matrix