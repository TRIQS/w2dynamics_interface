#!/usr/bin/env python

import sys, os, argparse
from itertools import product
from numpy import matrix, array, diag

from h5 import HDFArchive
from triqs.utility import mpi
from triqs.gf import Gf, MeshImFreq, iOmega_n, inverse, BlockGf, Fourier
from triqs.operators import c, c_dag, n
from triqs.operators.util.hamiltonians import h_int_kanamori

parser = argparse.ArgumentParser(description="Test arguments")
parser.add_argument('--libcxx', action='store_true', help="Use libcxx reference data")
args, unknown = parser.parse_known_args()

# ==== System Parameters ====
beta = 5.           # Inverse temperature
mu = 2.             # Chemical potential
U = 5.              # On-site density-density interaction
h = 0.2             # Local magnetic field
E = [ 0.0, 4.0 ]    # Bath-site energies
V = [ 2.0, 5.0 ]    # Couplings to Bath-sites

spin_names = ['up', 'dn']
orb_names  = [0]

# ==== Local Hamiltonian ====
h_0 = - mu*( n('up',0) + n('dn',0) ) - h*( n('up',0) - n('dn',0) )
h_int = U * n('up',0) * n('dn',0)
h_loc = h_0 + h_int

# ==== Bath & Coupling Hamiltonian ====
h_bath, h_coup = 0, 0
for i, E_i, V_i in zip([0, 1], E, V):
    for sig in ['up','dn']:
        h_bath += E_i * n(sig,'b_' + str(i))
        h_coup += V_i * (c_dag(sig,0) * c(sig,'b_' + str(i)) + c_dag(sig,'b_' + str(i)) * c(sig,0))

# ==== Total impurity hamiltonian and fundamental operators ====
h_imp = h_loc + h_coup + h_bath

# ==== Green function structure ====
gf_struct = [ [s, orb_names] for s in spin_names ]

# ==== Hybridization Function ====
n_iw = int(10 * beta)
iw_mesh = MeshImFreq(beta, 'Fermion', n_iw)
Delta = BlockGf(mesh=iw_mesh, gf_struct=gf_struct)
Delta << sum([V_i*V_i * inverse(iOmega_n - E_i) for V_i,E_i in zip(V, E)]);

# ==== Non-Interacting Impurity Green function  ====
G0_iw = BlockGf(mesh=iw_mesh, gf_struct=gf_struct)
G0_iw['up'] << inverse(iOmega_n + mu + h - Delta['up'])
G0_iw['dn'] << inverse(iOmega_n + mu - h - Delta['dn'])


# ==== Construct the CTHYB solver using the G0_iw Interface ====
constr_params = {
        'beta' : beta,
        'gf_struct' : gf_struct,
        'n_iw' : n_iw,
        'n_tau' : 1000
        }
from w2dyn_cthyb import Solver
S = Solver(**constr_params)

# --------- Initialize G0_iw ----------
S.G0_iw << G0_iw

# --------- Solve! ----------
solve_params = {
        'h_int' : h_int,
        'n_warmup_cycles' : 100,
        'n_cycles' : 5000,
        'length_cycle' : 100
        }
S.solve(**solve_params)

# -------- Save in archive ---------
if mpi.is_master_node():
    with HDFArchive("SIAM_Discrete_Bath.out.h5",'w') as results:
        results["G_iw"] = S.G_iw
        results["G_tau"] = S.G_tau

from triqs.utility.h5diff import h5diff
if args.libcxx:
    h5diff("SIAM_Discrete_Bath.libcxx.ref.h5","SIAM_Discrete_Bath.out.h5")
else:
    h5diff("SIAM_Discrete_Bath.ref.h5","SIAM_Discrete_Bath.out.h5")

# ==== Construct the CTHYB solver using the Delta_tau + h_0 Interface ====
constr_params = {
        'beta' : beta,
        'gf_struct' : gf_struct,
        'n_iw' : n_iw,
        'n_tau' : 1000,
        'delta_interface' : True
        }
S = Solver(**constr_params)

# --------- Initialize G0_iw ----------
S.Delta_tau << Fourier(Delta)

# --------- Solve! ----------
solve_params = {
        'h_int' : h_int,
        'n_warmup_cycles' : 100,
        'n_cycles' : 5000,
        'length_cycle' : 100,
        'h_0' : h_0
        }
S.solve(**solve_params)

# -------- Save in archive ---------
if mpi.is_master_node():
    with HDFArchive("SIAM_Discrete_Bath.delta_interface.out.h5",'w') as results:
        results["G_iw"] = S.G_iw
        results["G_tau"] = S.G_tau

if args.libcxx:
    h5diff("SIAM_Discrete_Bath.libcxx.ref.h5","SIAM_Discrete_Bath.delta_interface.out.h5")
else:
    h5diff("SIAM_Discrete_Bath.ref.h5","SIAM_Discrete_Bath.delta_interface.out.h5")
