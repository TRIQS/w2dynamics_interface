#!/usr/bin/env python

# Check that the improved estimators reproduce the Green's function of the
# Dyson equation for a single orbital, where all of them are exact.

from triqs.gf import BlockGf, MeshImFreq, inverse, iOmega_n
from triqs.operators import n
from triqs.utility.comparison_tests import assert_block_gfs_are_close

from w2dyn_cthyb import Solver

# ==== System Parameters ====
beta = 5.           # Inverse temperature
mu = 2.             # Chemical potential
U = 5.              # On-site density-density interaction
h = 0.2             # Local magnetic field
E = [ 0.0, 4.0 ]    # Bath-site energies
V = [ 2.0, 5.0 ]    # Couplings to Bath-sites

spin_names = ['up', 'dn']
gf_struct = [ [s, 1] for s in spin_names ]

h_int = U * n('up',0) * n('dn',0)

n_iw = int(10 * beta)
iw_mesh = MeshImFreq(beta, 'Fermion', n_iw)
Delta = BlockGf(mesh=iw_mesh, gf_struct=gf_struct)
Delta << sum([V_i*V_i * inverse(iOmega_n - E_i) for V_i,E_i in zip(V, E)])

G0_iw = BlockGf(mesh=iw_mesh, gf_struct=gf_struct)
G0_iw['up'] << inverse(iOmega_n + mu + h - Delta['up'])
G0_iw['dn'] << inverse(iOmega_n + mu - h - Delta['dn'])

# ==== Solve for every estimator ====
G_iw = {}
for selfenergy in ['dyson', 'improved_worm', 'symmetric_improved_worm']:
    S = Solver(beta=beta, gf_struct=gf_struct, n_iw=n_iw, n_tau=1000)
    S.G0_iw << G0_iw
    S.solve(h_int=h_int, n_warmup_cycles=1000, n_cycles=50000, length_cycle=50,
            selfenergy=selfenergy)

    # The improved estimators sample a worm sector that does not contain the
    # Green's function in imaginary time
    if selfenergy == 'dyson':
        assert hasattr(S, 'G_tau')
    else:
        assert not hasattr(S, 'G_tau')

    G_iw[selfenergy] = S.G_iw.copy()

for selfenergy in ['improved_worm', 'symmetric_improved_worm']:
    assert_block_gfs_are_close(G_iw[selfenergy], G_iw['dyson'], precision=2.e-2)
