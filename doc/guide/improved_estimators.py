from triqs.gf import *
from triqs.operators import *
from h5 import HDFArchive
import triqs.utility.mpi as mpi

from w2dyn_cthyb import Solver

# Parameters: one impurity orbital in a magnetic field, coupled to two bath sites
U, mu, h, beta = 5.0, 2.0, 0.2, 5.0
E, V = [0.0, 4.0], [2.0, 5.0]

gf_struct = [('up', 1), ('dn', 1)]
n_iw = int(10 * beta)

# Hybridization function of the two bath sites
Delta = BlockGf(mesh=MeshImFreq(beta, 'Fermion', n_iw), gf_struct=gf_struct)
Delta << sum([V_i**2 * inverse(iOmega_n - E_i) for V_i, E_i in zip(V, E)])

# Solve the same model with every estimator for the self-energy
results = {}
for selfenergy in ['dyson', 'improved_worm', 'symmetric_improved_worm']:

    S = Solver(beta=beta, gf_struct=gf_struct, n_iw=n_iw, n_tau=1000)

    S.G0_iw['up'] << inverse(iOmega_n + mu + h - Delta['up'])
    S.G0_iw['dn'] << inverse(iOmega_n + mu - h - Delta['dn'])

    S.solve(h_int=U * n('up', 0) * n('dn', 0),  # Local Hamiltonian
            n_cycles=500000,                    # Number of QMC cycles
            length_cycle=50,                    # Length of one cycle
            n_warmup_cycles=1000,               # Warmup cycles
            selfenergy=selfenergy)              # Estimator for the self-energy

    results[selfenergy] = {"G_iw": S.G_iw, "Sigma_iw": S.Sigma_iw,
                           "Sigma_moments": S.Sigma_moments}

# Save the results in an HDF5 file (only on the master node)
if mpi.is_master_node():
    with HDFArchive("improved_estimators_solution.h5", 'w') as Results:
        for selfenergy, result in results.items():
            Results[selfenergy] = result
