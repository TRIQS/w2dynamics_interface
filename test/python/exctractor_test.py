#!/usr/bin/env python

from w2dyn_cthyb.converters import *
from w2dyn_cthyb.extractor import *
from pytriqs.gf import MeshImFreq
from pytriqs.gf import BlockGf, inverse, iOmega_n, Fourier

import numpy as np

### generate a test-impurity
ntau = 1000
niw = 2000
beta = 20.0
norb = 5

### generate block structure
spin_names = ['up', 'dn']
orb_names  = [ i for i in range(0,norb)]
gf_struct = [ [s, orb_names] for s in spin_names ]

### generate hybridisation function
iw_mesh = MeshImFreq(beta, 'Fermion', niw)
Delta_iw = BlockGf(mesh=iw_mesh, gf_struct=gf_struct)

### generate random values for hybridisation function
for block, delta_iw in Delta_iw:

    ### generate random hermitian bath energies
    s = (norb , norb )
    eps = 5.0 * (np.random.random(s) - 0.5) + 3.j * (np.random.random(s) - 0.5)
    eps = eps + np.conjugate(eps.T)

    delta_block = inverse(iOmega_n - eps)
    delta_iw << delta_block

    ### multiply everything (element-wise) with hermitian matrix
    ### since Delta(iw) does not have high frequency tail of 1
    randmat = 5.0 * (np.random.random(s) - 0.5) + 3.j * (np.random.random(s) - 0.5)
    randmat = randmat + np.conjugate(randmat.T)

    for ni,i in enumerate(delta_iw.data):
        delta_iw.data[ni,:,:,] = np.multiply(i, randmat)

### generate G0
G0_iw = BlockGf(mesh=iw_mesh, gf_struct=gf_struct)
H_0_original = []

for block, g0_iw in G0_iw:

    ### random hermitian h_0
    s = (norb, norb)
    tmp = -1.0 * (np.random.random(s) - 0.5) + 2.j * (np.random.random(s) - 0.5)
    h_0 = tmp + np.conjugate(tmp.T)
    g0_iw << inverse( iOmega_n - h_0 - Delta_iw[block] )

    H_0_original.append(h_0)

### extract Delta and hopping matrix from G0 with function
Delta_iw_reconst, H_0_reconst = extract_deltaiw_and_tij_from_G0(G0_iw, gf_struct)

### compare Delta
from pytriqs.utility.comparison_tests import *
assert_block_gfs_are_close(Delta_iw_reconst, Delta_iw)

### compare H_0
for i,j in zip(H_0_reconst, H_0_original):
    np.testing.assert_almost_equal(i,j)
    # print 'i-j', i-j
