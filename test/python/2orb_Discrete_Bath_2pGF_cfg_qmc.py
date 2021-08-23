#!/usr/bin/env python

import itertools
import numpy as np

import sys, os, argparse
from itertools import product
from numpy import matrix, array, diag

from h5 import HDFArchive
from triqs.utility import mpi
from triqs.gf import Gf, MeshImFreq, iOmega_n, inverse, BlockGf, Fourier, Block2Gf
from triqs.operators import c, c_dag, n
from triqs.operators.util.hamiltonians import h_int_kanamori
from itertools import product
from numpy import matrix, array, diag, eye
from numpy.linalg import inv

parser = argparse.ArgumentParser(description="Test arguments")
parser.add_argument('--libcxx', action='store_true', help="Use libcxx reference data")
parser.add_argument('--gccver_ge11', action='store_true', help="Use gcc11+ reference data")
args, unknown = parser.parse_known_args()

# ==== System Parameters ====
no = 2                          # Number of orbitals
beta = 30.                      # Inverse temperature
mu = 3.0                        # Chemical potential
eps = array([-0.2, 0.3])         # Impurity site energies
t = 0.2                         # Hopping between impurity sites

eps_bath = array([0.12, -0.17])  # Bath site energies

U = 2.                          # On-site interaction
V = 1.                          # Intersite interaction
J = 0.5                         # Hunds coupling

spin_names = ['up', 'dn']
orb_names = [0, 1]
so_names = ['_'.join((s, str(o))) for s in spin_names for o in orb_names]
orb_bath_names = [0, 1]

# Non-interacting impurity hamiltonian in matrix representation
h_0_mat = diag(eps - mu)

# Bath hamiltonian in matrix representation
h_bath_mat = diag(eps_bath)

# Coupling matrix
# V_mat = matrix([[1., 0.],
#                 [0., 1.]])

# ==== Local Hamiltonian ====
c_dag_vec = {bl: matrix([[c_dag(bl, 0)]]) for bl in so_names}
c_vec = {bl: matrix([[c(bl, 0)]]) for bl in so_names}

h_0 = sum(c_dag_vec[bl] * diag(h_0_mat)[i % no] * c_vec[bl] for i, bl in enumerate(so_names))[0, 0]

h_int = h_int_kanamori(spin_names, orb_names,
                       array([[0, V-J],
                              [V-J, 0]]),  # Interaction for equal spins
                       array([[U, V],
                              [V, U]]),   # Interaction for opposite spins
                       J, False)

h_loc = h_0 + h_int

# ==== Bath & Coupling hamiltonian ====
so_bath_names = ['b_' + so for so in so_names]
c_dag_bath_vec = {bl: matrix([[c_dag(bl, 0)]]) for bl in so_names}
c_bath_vec = {bl: matrix([[c(bl, 0)]]) for bl in so_names}

h_bath = sum(c_dag_bath_vec[bl] * diag(h_bath_mat)[i % no] * c_bath_vec[bl]
             for i, bl in enumerate(so_names))[0, 0]
h_coup = sum((c_dag_vec[s] * c_bath_vec[s]
              + c_dag_bath_vec[s] * c_vec[s])
             for s in so_names)[0, 0]  # FIXME Adjoint

# ==== Total impurity hamiltonian ====
h_imp = h_loc + h_coup + h_bath

# ==== Green function structure ====
gf_struct = [[so, 1] for so in so_names]

# ==== Hybridization Function ====
n_iw = int(10 * beta)
iw_mesh = MeshImFreq(beta, 'Fermion', n_iw)
Delta = BlockGf(mesh=iw_mesh, gf_struct=gf_struct)
for i, (bl, iw) in enumerate(product(so_names, iw_mesh)):
    Delta[bl][iw] = inv(iw.value * eye(1) - diag(h_bath_mat)[i % no])

# ==== Non-Interacting Impurity Green function  ====
G0_iw = Delta.copy()
for i, bl in enumerate(so_names):
    G0_iw[bl] << inverse(iOmega_n - diag(h_0_mat)[i % no] - Delta[bl])


# ==== Construct the CTHYB solver using the G0_iw Interface ====
constr_params = {
        'beta' : beta,
        'gf_struct' : gf_struct,
        'n_iw' : n_iw,
        'n_tau' : 10000
        }
from w2dyn_cthyb import Solver
S = Solver(**constr_params)

# --------- Initialize G0_iw ----------
S.G0_iw << G0_iw

# --------- Solve! ----------

# check only some components, both ones expected to be zero and nonzero
worm_components = np.array([1, 2, 6, 11, 16, 21, 31, 35, 40, 41, 46, 55, 61])

# convert worm components to cfg_qmc["WormComponents"] compatible format
WormComponents = np.array([ list(x) for x in itertools.product(range(4), repeat=4) ], dtype=object)
WormComponents = WormComponents[worm_components - 1] # Select just the few components with indices given above

# Expert user mode, pass `cfg_qmc` dictionary to solver in W2Dynamics internal format

solve_params = {
    'h_int': h_int,
    'n_warmup_cycles': 10000,
    'n_cycles': 1000,
    'length_cycle': 20,
    'worm' : True,
    'cfg_qmc' : dict(
        WormEta=1,
        WormSearchEta=1,
        WormMeasG4iw=1,
        FourPnt=8,
        N4iwb=4,
        N4iwf=5,
        WormComponents=WormComponents,
        ),
}

S.solve(**solve_params)

# -- Parse worm components to TPRF compatible tensor two particle Green's function
from w2dyn_cthyb.converters_worm import g2_from_w2dyn_G2_worm_components
g2 = g2_from_w2dyn_G2_worm_components(S.G2_worm_components, 2)

# -- To compare with the reference data we have to
# -- convert to (somewhat arcane) block format used in the old G2 measure

blocks = []
for i1, (b1, s1) in enumerate(gf_struct):
    sub_blocks = []
    for i2, (b2, s2) in enumerate(gf_struct):
        g_tensor = Gf(mesh=g2.mesh, target_shape=[1]*4)
        g_tensor.data[..., 0, 0, 0, 0] = g2[i1, i1, i2, i2].data
        sub_blocks.append(g_tensor)
    blocks.append(sub_blocks)

block_names = [x[0] for x in gf_struct]
S.G2_iw_ph = Block2Gf(block_names, block_names, blocks)

# -------- Save in archive ---------
if mpi.is_master_node():
    with HDFArchive("2orb_Discrete_Bath_2pGF_cft_qmc.out.h5", 'w') as results:
        results["G2_iw_ph"] = S.G2_iw_ph

from triqs.utility.h5diff import h5diff
if args.libcxx:
    h5diff("2orb_Discrete_Bath_2pGF.libcxx.ref.h5",
           "2orb_Discrete_Bath_2pGF_cft_qmc.out.h5",
           precision=1.e-5)
elif args.gccver_ge11:
    h5diff("2orb_Discrete_Bath_2pGF.gccver_ge11.ref.h5",
           "2orb_Discrete_Bath_2pGF_cft_qmc.out.h5",
           precision=1.e-5)
else:
    h5diff("2orb_Discrete_Bath_2pGF.ref.h5",
           "2orb_Discrete_Bath_2pGF_cft_qmc.out.h5",
           precision=1.e-5)
