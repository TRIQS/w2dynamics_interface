################################################################################
#
# w2dynamics_interface - An Interface to the w2dynamics cthyb code
#
# Copyright (C) 2032 by Hugo U. R. Strand, Erik van Loon
# Authors: Hugo U.R. Strand, Erik van Loon
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
# TPRF. If not, see <http://www.gnu.org/licenses/>.
#
################################################################################

import numpy as np

from triqs.gf import Gf, MeshProduct, Idx

from w2dyn.auxiliaries.compound_index import index2component_general

    
def g2_from_w2dyn_G2_worm_components(G2_worm_components, Nbands, so_tr=True):

    """Get TRIQS two-particle Green's function
    from W2Dynamics worm sampled components."""
    
    _, g2, _ = G2_worm_components[0]

    fmesh, _, bmesh = g2.mesh.components
    mesh = MeshProduct(bmesh, fmesh, fmesh)
    beta = bmesh.beta
    
    target_shape = tuple([2*Nbands]*4)

    G2 = Gf(mesh=mesh, target_shape=target_shape)
    
    for index, g2_wwn, g2_wnn_err in G2_worm_components:
        bs, b, s = index2component_general(Nbands, 4, index)        
        i, j, k, l = [ int(x) for x in bs ]        
        G2[j, i, l, k].data[:] = beta * np.moveaxis(g2_wwn.data, -1, 0)

    if so_tr: G2 = transpose_rank4_gfs_from_orbspin_to_spinorb(G2)
        
    return G2


def p3_from_w2dyn_P3_worm_components(P3_worm_components, Nbands, so_tr=True):

    """Get TRIQS three-point Green's function
    from W2Dynamics worm sampled components."""

    _, p3, _ = P3_worm_components[0]

    fmesh, bmesh = p3.mesh.components
    mesh = MeshProduct(bmesh, fmesh)
    beta = bmesh.beta
    
    target_shape = tuple([2*Nbands]*4)

    P3 = Gf(mesh=mesh, target_shape=target_shape)
    
    for index, p3_wn, p3_wn_err in P3_worm_components:
        bs, b, s = index2component_general(Nbands, 4, index)        
        i, j, k, l = [ int(x) for x in bs ] 
        P3[j, i, l, k].data[:] = beta * np.moveaxis(p3_wn.data, -1, 0)

    if so_tr: P3 = transpose_rank4_gfs_from_orbspin_to_spinorb(P3)

    return P3


def p2_from_w2dyn_P2_worm_components(P2_worm_components, Nbands, so_tr=True):

    """Get TRIQS two-particle susceptibility Green's function
    from W2Dynamics worm sampled components."""

    _, p2, _ = P2_worm_components[0]

    bmesh = p2.mesh
    mesh = bmesh
    beta = bmesh.beta
    
    target_shape = tuple([2*Nbands]*4)

    P2 = Gf(mesh=mesh, target_shape=target_shape)
    
    for index, p2_w, p2_w_err in P2_worm_components:
        bs, b, s = index2component_general(Nbands, 4, index)        
        i, j, k, l = [ int(x) for x in bs ] 
        P2[j, i, l, k].data[:] = p2_w.data

    if so_tr: P2 = transpose_rank4_gfs_from_orbspin_to_spinorb(P2)

    return P2


def p2_remove_disconnected(p2,Gtau):

    """Remove disconnected part of p2"""
    
    p2_conn = p2.copy()
    beta = p2.mesh.beta
    
    # Only at zero frequency
    p2_conn[Idx(0)] -= beta*np.einsum('ab,cd->abcd', Gtau(0), Gtau(0))

    return p2_conn


def p3_w2dyn_to_triqs_freq_shift(p3):

    """Perform a fermionic frequency shift passing from the W2Dynamics notation
    to the TRIQS/TPRF frequency convention. """
    
    p3_new = Gf(mesh=p3.mesh, target_shape=p3.target_shape)
    
    def in_mesh(nu):
        return p3.mesh[1].first_index() <= nu_shifted_index <= p3.mesh[1].last_index()
        
    for omega,nu in p3.mesh:
        nu_shifted_index = nu.index - omega.index
        if not in_mesh(nu_shifted_index):
            continue
        p3_new[omega,Idx(nu_shifted_index)] = p3[omega,nu]
    
    return p3_new


def p3_w2dyn_to_triqs_freq_shift_alt(p3):
    
    """Perform a fermionic frequency shift passing from the W2Dynamics notation
    to the TRIQS/TPRF frequency convention.
    
    This alternative implementation uses the time-reversal symmetry of the action."""
    
    p3_new = Gf(mesh=p3.mesh,target_shape=p3.target_shape)
    for omega,nu in p3.mesh:
        p3_new[Idx(-omega.index),nu] = np.einsum('abcd->badc',p3[omega,nu])
    return p3_new


def L_from_g3(g3, G_w, return_chi0_inv=False):
    
    """Construct the triangle vertex L from g3

    L = (g3 - \beta * G * G) (GG)^{-1}
    
    by removing the disconnected part (\beta * G * G) from g3 and
    then amputating the fermionic legs with G, using (\chi_0^{-1} = (GG)^{-1})
    """

    bmesh, fmesh = g3.mesh[0], g3.mesh[1]

    g = Gf(mesh=fmesh, target_shape=G_w.target_shape)
    for nu in fmesh: g[nu] = G_w(nu.index) # Truncate g to fmesh of g3

    G_tau_0 = -np.eye(G_w.target_shape[0]) + G_w.density() # G(tau=0) from G(iw)

    # Remove disconnected terms (at zero bosonic frequency)
    g3_conn = g3.copy()
    g3_conn[Idx(0), :].data[:] -= fmesh.beta * np.einsum('nba,dc->nabcd', g.data, G_tau_0)

    # Construct gg bubble inverse
    from triqs_tprf.linalg import inverse_PH
    from triqs_tprf.chi_from_gg2 import chi0_from_gg2_PH
    tmp = Gf(mesh=MeshProduct(bmesh, fmesh, fmesh), target_shape=g3.target_shape)
    chi0_inv = inverse_PH(chi0_from_gg2_PH(G_w, tmp))

    # Amputate gg bubble
    L = g3.copy() 
    L.data[:] = np.einsum('wnefcd,wnnabfe->wnabcd', g3_conn.data, chi0_inv.data)

    if return_chi0_inv:
        return L, chi0_inv
    else:
        return L


def transpose_rank4_gfs_from_orbspin_to_spinorb(g):
    
    """ Rearrange tensor indices in A Greens function
    with rank 4 target space `len(g.target_shape) == 4`
    W2Dynamics orders orbital indices and then spin indices
    while in TRIQS we have the opposite, spins then orbitals.
    """

    data = g.data.copy()
    
    s_pre, s_aaaa = data.shape[:-4], data.shape[-4:]

    na = s_aaaa[0]
    no = na // 2
    s_ososos = [no, 2] * 4
    shape_os = list(s_pre) + s_ososos

    perm = len(s_pre) + np.array([1, 0, 3, 2, 5, 4, 7, 6])
    axes = list(range(len(s_pre))) + list(perm)

    data = np.reshape(data, shape_os)
    data = np.transpose(data, axes=axes)

    g.data[:] = np.reshape(data, g.data.shape)

    return g
