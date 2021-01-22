#!/usr/bin/env python

from triqs.gf import *
from triqs.operators import *
from h5 import HDFArchive
import triqs.utility.mpi as mpi
import numpy as np

#w2dyn=False
w2dyn=True

### here comes the solver
if w2dyn:
   from w2dyn_cthyb import Solver
else:
   from triqs_cthyb import Solver

from w2dyn_cthyb.converters import *

### Parameters
U = 1.0
beta = 100.0
#e_f = 0.5
e_f = -U/2.0

### a discrete AIM
e1, e2 = -0.43, 0.134
V1, V2 = 0.38, 0.38

### Construct the impurity solver with the inverse temperature
### and the structure of the Green's functions
S = Solver(beta = beta, gf_struct = [ ['up',1], ['down',1] ], n_iw=2000,  n_tau=4002)

### the hybridistation function in Matsubara
delta_iw_block = ( V1**2 * inverse( iOmega_n - e1 ) + V2**2 * inverse( iOmega_n - e2 ) )

### Initialize the non-interacting Green's function S.G0_iw
for name, g0 in S.G0_iw: 
    g0 << inverse ( iOmega_n - e_f - delta_iw_block )

### Run the solver. The results will be in S.G_tau, S.G_iw and S.G_l
if w2dyn:
    S.solve(
        #h_int = U * n('up',0) * n('down',0) - e_f* ( n('up',0) + n('down',0) ),     # Local Hamiltonian + quadratic terms
        h_int = U * n('up',0) * n('down',0),     # Local Hamiltonian + quadratic terms
        n_cycles  = 5000,                      # Number of QMC cycles
        length_cycle = 100,                      # Length of one cycle
        n_warmup_cycles = 5000,                 # Warmup cycles
        measure_G_l = False)                      # Measure G_l
else:
    S.solve(h_int = U * n('up',0) * n('down',0) ,     # Local Hamiltonian
            n_cycles  = 5000,                      # Number of QMC cycles
            length_cycle = 100,                      # Length of one cycle
            n_warmup_cycles = 5000,                 # Warmup cycles
            measure_G_l = False)                      # Measure G_l

### plot Greens function
#from triqs.plot.mpl_interface import oplot, oploti, oplotr, plt
#oplot(S.G_tau)
#plt.show()

### write stuff to file
if w2dyn:
   with HDFArchive("aim_solution_w2dyn.h5",'w') as Results:
       Results["G_tau"] = S.G_tau
       Results["G_iw"] = S.G_iw
else:
   with HDFArchive("aim_solution_triqs.h5",'w') as Results:
       Results["G_tau"] = S.G_tau
       Results["G_iw"] = S.G_iw
