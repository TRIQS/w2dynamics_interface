#!/usr/bin/env python

import sys, os, argparse
from itertools import product
from numpy import matrix, array, diag

from h5 import HDFArchive
from triqs.utility import mpi
from triqs.gf import Gf, MeshImFreq, iOmega_n, inverse, BlockGf, Fourier
from triqs.operators import c, c_dag, n
from triqs.operators.util.hamiltonians import h_int_kanamori
from itertools import product
from numpy import matrix, array, diag, eye
from numpy.linalg import inv

parser = argparse.ArgumentParser(description="Test arguments")
parser.add_argument('--libcxx', action='store_true', help="Use libcxx reference data")
args, unknown = parser.parse_known_args()

# ==== System Parameters ====
beta = 30.                      # Inverse temperature
mu = 3.0                        # Chemical potential
eps = array([-0.2, 0.3])         # Impurity site energies
t = 0.2                         # Hopping between impurity sites

eps_bath = array([-0.4, 0.12, -0.17, 0.65])  # Bath site energies

U = 2.                          # On-site interaction
V = 1.                          # Intersite interaction
J = 0.5                         # Hunds coupling

spin_names = ['up', 'dn']
orb_names  = [0, 1]
orb_bath_names  = [0, 1, 2, 3]

# Non-interacting impurity hamiltonian in matrix representation
h_0_mat = diag(eps - mu) - matrix([[0, t],
                                   [t, 0]])

# Bath hamiltonian in matrix representation
h_bath_mat = diag(eps_bath)

# Coupling matrix
V_mat = matrix([[1., 1., 0.2, 0.2],
                [0.2, 0.2, 1, 1]])

# ==== Local Hamiltonian ====
c_dag_vec = { s: matrix([[c_dag(s,o) for o in orb_names]]) for s in spin_names }
c_vec =     { s: matrix([[c(s,o)] for o in orb_names]) for s in spin_names }

h_0 = sum(c_dag_vec[s] * h_0_mat * c_vec[s] for s in spin_names)[0,0]

h_int = h_int_kanamori(spin_names, orb_names,
                        array([[0,      V-J ],
                               [V-J   , 0   ]]), # Interaction for equal spins
                        array([[U,      V ],
                               [V,      U ]]),   # Interaction for opposite spins
                        J,True)

h_loc = h_0 + h_int

# ==== Bath & Coupling hamiltonian ====
orb_bath_names = ['b_' + str(o) for o in orb_bath_names]
c_dag_bath_vec = { s: matrix([[c_dag(s, o) for o in orb_bath_names]]) for s in spin_names }
c_bath_vec =     { s: matrix([[c(s, o)] for o in orb_bath_names]) for s in spin_names }

h_bath = sum(c_dag_bath_vec[s] * h_bath_mat * c_bath_vec[s] for s in spin_names)[0,0]
h_coup = sum(c_dag_vec[s] * V_mat * c_bath_vec[s] + c_dag_bath_vec[s] * V_mat.transpose() * c_vec[s] for s in spin_names)[0,0] # FIXME Adjoint

# ==== Total impurity hamiltonian ====
h_imp = h_loc + h_coup + h_bath

# ==== Green function structure ====
gf_struct = [ [s, orb_names] for s in spin_names ]

# ==== Hybridization Function ====
n_iw = int(10 * beta)
iw_mesh = MeshImFreq(beta, 'Fermion', n_iw)
Delta = BlockGf(mesh=iw_mesh, gf_struct=gf_struct)
# FIXME Delta['up'] << V_mat.transpose() * inverse(iOmega_n - h_bath_mat) * V_mat
for bl, iw in product(spin_names, iw_mesh):
    Delta[bl][iw] = V_mat * inv(iw.value * eye(len(orb_bath_names)) - h_bath_mat) * V_mat.transpose()

# ==== Non-Interacting Impurity Green function  ====
G0_iw = Delta.copy()
G0_iw['up'] << inverse(iOmega_n - h_0_mat - Delta['up']) # FIXME Should work for BlockGf
G0_iw['dn'] << inverse(iOmega_n - h_0_mat - Delta['dn'])


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
solve_params = {
        'h_int' : h_int,
        'n_warmup_cycles' : 100,
        'n_cycles' : 1000,
        'length_cycle' : 100
        }
S.solve(**solve_params)

# -------- Save in archive ---------
if mpi.is_master_node():
    with HDFArchive("2orb_Discrete_Bath.out.h5",'w') as results:
        results["G_iw"] = S.G_iw
        results["G_tau"] = S.G_tau

from triqs.utility.h5diff import h5diff
if args.libcxx:
    h5diff("2orb_Discrete_Bath.libcxx.ref.h5","2orb_Discrete_Bath.out.h5", precision=1.e-5)
else:
    h5diff("2orb_Discrete_Bath.ref.h5","2orb_Discrete_Bath.out.h5", precision=1.e-5)


# ==== Construct the CTHYB solver using the Delta_tau + h_0 Interface ====
constr_params = {
        'beta' : beta,
        'gf_struct' : gf_struct,
        'n_iw' : n_iw,
        'n_tau' : 10000,
        'delta_interface' : True
        }
S = Solver(**constr_params)

# --------- Initialize Delta_iw ----------
S.Delta_tau << Fourier(Delta)

# --------- Solve! ----------
solve_params = {
        'h_int' : h_int,
        'n_warmup_cycles' : 100,
        'n_cycles' : 1000,
        'length_cycle' : 100,
        'h_0' : h_0
        }
S.solve(**solve_params)

# -------- Save in archive ---------
if mpi.is_master_node():
    with HDFArchive("2orb_Discrete_Bath.delta_interface.out.h5",'w') as results:
        results["G_iw"] = S.G_iw
        results["G_tau"] = S.G_tau

if args.libcxx:
    h5diff("2orb_Discrete_Bath.libcxx.ref.h5", "2orb_Discrete_Bath.delta_interface.out.h5",
       precision=1.e-3)
else:
    h5diff("2orb_Discrete_Bath.ref.h5", "2orb_Discrete_Bath.delta_interface.out.h5",
       precision=1.e-3)
