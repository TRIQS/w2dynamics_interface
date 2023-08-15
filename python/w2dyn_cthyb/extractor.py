################################################################################
#
# w2dynamics_interface - An Interface to the w2dynamics cthyb code
#
# Copyright (C) 2019 by The Simons Foundation
# Authors: Andreas Hausoel, Hugo U. R. Strand, Nils Wentzell
#
# w2dynamics_interface is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# w2dynamics_interface is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# w2dynamics_interface. If not, see <http://www.gnu.org/licenses/>.
#
################################################################################

import numpy as np
from .converters import *
from triqs.gf import MeshImFreq
from triqs.gf import BlockGf, inverse, iOmega_n, Fourier

# ----------------------------------------------------------------------    
def extract_deltaiw_and_tij_from_G0(G0_iw, gf_struct):

    iw_mesh = G0_iw.mesh
    Delta_iw= BlockGf(mesh=iw_mesh, gf_struct=gf_struct)
    H_loc_block = []

    for block, g0_iw in G0_iw:

        tail, err = g0_iw.fit_hermitian_tail()
        #print "---------------"
        #print "tail", tail
        H_loc = tail[2]
        #print "---------------"
        #print "H_loc", H_loc
        Delta_iw[block] << iOmega_n - H_loc - inverse(g0_iw)

        H_loc_block.append(H_loc)

    return Delta_iw, H_loc_block


# ----------------------------------------------------------------------    
### test program for extractor
if __name__ == '__main__':
    
    ### generate a test-impurity
    gf_struct, Delta_tau, H_loc = get_test_impurity_model(norb=3, ntau=1000, beta=10.0)

    beta = Delta_tau.mesh.beta
    n_iw = 100
    n_tau = 1000


    ### generate a hybridisation function
    iw_mesh = MeshImFreq(beta, 'Fermion', n_iw)
    Delta_iw = BlockGf(mesh=iw_mesh, gf_struct=gf_struct)

    for block, delta_iw in Delta_iw:
        s = (3, 3)
        eps = np.random.random(s) + 1.j * np.random.random(s)
        eps = eps + np.conjugate(eps.T)

        delta_block = inverse(iOmega_n - 0.3 * eps)
        delta_iw << delta_block

    
    ### generate the corresponding hybridisation function and hopping matrix 
    G0_iw = BlockGf(mesh=iw_mesh, gf_struct=gf_struct)
    H_loc_original = []

    for block, g0_iw in G0_iw:
        #print "block", block
        s = (3, 3)
        H = np.random.random(s) + 1.j * np.random.random(s)
        H = H + np.conjugate(H.T)
        #print ""
        #print "H", H
        g0_iw << inverse( iOmega_n - H - Delta_iw[block] )

        H_loc_original.append(H)

    ### extract Delta and hopping matrix from G0 with function
    Delta_iw_reconst, H_loc_reconst = extract_deltaiw_and_tij_from_G0(G0_iw, gf_struct)

    ### extract Delta and hopping matrix from G0 (alternative)
    #Delta_iw_reconst = BlockGf(mesh=iw_mesh, gf_struct=gf_struct)
    #H_loc_reconst = []
    
    #for block, g0_iw in G0_iw:

        #tail, err = g0_iw.fit_hermitian_tail()
        ##print "---------------"
        ##print "tail", tail
        #H_loc = tail[2]
        #print "---------------"
        #print "H_loc", H_loc
        #Delta_iw_reconst[block] << iOmega_n - H_loc - inverse(g0_iw)

        #H_loc_reconst.append(H_loc)
        
    
    # ------------------------------------------------------------------
    print(" ")
    print("compare H_loc:")
    for i,j in zip(H_loc_original,H_loc_reconst):
        print("np.max(i-j)", np.max(i-j))
        print("np.min(i-j)", np.min(i-j))

    print(" ")
    print("compare delta:")
    
    for block, _ in Delta_iw:
        d1 = Delta_iw[block].data
        d2 = Delta_iw_reconst[block].data

        #np.testing.assert_array_almost_equal(d1,d2)

        print("np.amax(np.real(d1-d2))", np.amax(np.real(d1-d2)))
        print("np.amin(np.real(d1-d2))", np.amin(np.real(d1-d2)))
        print("np.amax(np.imag(d1-d2))", np.amax(np.imag(d1-d2)))
        print("np.amin(np.imag(d1-d2))", np.amin(np.imag(d1-d2)))

