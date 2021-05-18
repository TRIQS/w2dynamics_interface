
import numpy as np
    
# ----------------------------------------------------------------------    
def get_test_impurity_model(norb=2, ntau=1000, beta=10.0):

    """ Function that generates a random impurity model for testing """

    from triqs.operators import c, c_dag, Operator, dagger

    from pyed.OperatorUtils import fundamental_operators_from_gf_struct
    from pyed.OperatorUtils import symmetrize_quartic_tensor
    from pyed.OperatorUtils import get_quadratic_operator
    from pyed.OperatorUtils import operator_from_quartic_tensor
    
    spin_idxs = ['up', 'do']
    gf_struct = [ [spin_idx, norb] for spin_idx in spin_idxs ]

    # -- Random Hamiltonian
    
    fundamental_operators = fundamental_operators_from_gf_struct(gf_struct)
    #print "fundamental_operators ", fundamental_operators 

    N = len(fundamental_operators)
    t_OO = np.random.random((N, N)) + 1.j * np.random.random((N, N))
    t_OO = 0.5 * ( t_OO + np.conj(t_OO.T) )

    #print "N", N
    #print "t_OO", t_OO.shape


    #print 't_OO.real =\n', t_OO.real
    #print 't_OO.imag =\n', t_OO.imag
    
    U_OOOO = np.random.random((N, N, N, N)) + 1.j * np.random.random((N, N, N, N))
    U_OOOO = symmetrize_quartic_tensor(U_OOOO, conjugation=True)    
    
    #print 'gf_struct =', gf_struct
    #print 'fundamental_operators = ', fundamental_operators


    H_loc = get_quadratic_operator(t_OO, fundamental_operators) + \
        operator_from_quartic_tensor(U_OOOO, fundamental_operators)

    #print 'H_loc =', H_loc
    #print "H_loc.type", H_loc.type()

    from triqs.gf import MeshImTime, BlockGf

    mesh = MeshImTime(beta, 'Fermion', ntau)
    Delta_tau = BlockGf(mesh=mesh, gf_struct=gf_struct)

    #print "mesh", mesh
    #print "Delta_tau", Delta_tau

    for block_name, delta_tau in Delta_tau:
        delta_tau.data[:] = -0.5

    return gf_struct, Delta_tau, H_loc

# ----------------------------------------------------------------------    
def NO_to_Nos(A_NO, spin_first=False):

    """ Reshape a rank N tensor with composite spin and orbital
    index to a rank 2N tensor with orbital and then spin index.

    Default is that the composite index has the spin index as the first 
    (slow) index.

    If spin index is the fastest index set: spin_first=False 

    Author: Hugo U. R. Strand (2019)"""
    
    shape = A_NO.shape
    N = len(shape)
    norb = shape[-1] / 2

    assert( shape[-1] % 2 == 0 )
    np.testing.assert_array_almost_equal(
        np.array(shape) - shape[-1], np.zeros(N))
    
    if spin_first:
        shape_Nso = [2, norb] * N
        A_Nso = A_NO.reshape(shape_Nso)

        # Enumerate axes with every pair of indices permuted
        # i.e. N = 2 givex axes = [1, 0, 3, 2]
        axes = np.arange(2 * N).reshape((N, 2))[:, ::-1].flatten()
        
        A_Nos = np.transpose(A_Nso, axes=axes)
    else:
        shape_Nos = [norb, 2] * N
        A_Nos = A_NO.reshape(shape_Nos)

    return A_Nos

# ----------------------------------------------------------------------    
def triqs_gf_to_w2dyn_ndarray_g_tosos_beta_ntau(G_tau):

    """ Convert a spin-block Triqs imaginary time response function 
    (BlockGF) to  W2Dynamics ndarray format with indices [tosos]
    where t is time, o is orbital, s is spin index. 

    Returns: 
    g_tosos : ndarray with the response function data
    beta : inverse temperature
    ntau : number of tau points (including tau=0 and beta) 

    Author: Hugo U. R. Strand (2019) """

    ### test gf object
    full_size = 0
    for name, g in G_tau:
        size_block = g.data.shape[-1]
        full_size += size_block

    assert full_size % 2 == 0, "Number of flavours cannot be odd in w2dyn_cthyb!"

    beta = G_tau.mesh.beta
    tau = np.array([ float(t) for t in G_tau.mesh ])
    ntau = len(tau)
    np.testing.assert_almost_equal(tau[-1], beta)

    ### this will be the full object
    g_tff = np.zeros(shape=(ntau,full_size,full_size),dtype=g.data.dtype)

    ### writing the blocks in full object
    offset = 0
    for name, g in G_tau:
        size_block = g.data.shape[-1]
        g_tff[:,offset:offset+size_block,offset:offset+size_block] = g.data

        offset += size_block

    ### spin is slow running index, but in w2dyn it is fastest running index
    g_tosos = g_tff.reshape(ntau,2,full_size//2,2,full_size//2)
    g_tosos = g_tosos.transpose(0,2,1,4,3)

    return g_tosos, beta, ntau

# ----------------------------------------------------------------------    
def w2dyn_ndarray_to_triqs_BlockGF_tau_beta_ntau(gtau_osost, beta, gf_struct):

    """ Convert a DistributedSample of data in W2Dynamics
    ndarray format with indices [osost]
    where t is time, o is orbital, s is spin index
    to a spin-block Triqs imaginary time response function 

    Takes: 
    gtau : Green function as DistributedSample
    beta : inverse temperature
    ntau : number of tau points (including tau=0 and beta) 

    Returns: 
    BlockGF : triqs block Green function the response function data

    Author: Hugo U. R. Strand (2019) """

    gtau = gtau_osost.mean()
    gtau_err = gtau_osost.stderr()

    n_tau = gtau.shape[-1]
    assert n_tau % 2 == 0, "Need an even number of tau points to downsample to Triqs tau mesh"

    # -- Average over interior bins to simulate Triqs bin structure
    # -- with half-width edge bins.

    def average_interior_bins(gtau):
        
        gtau_0 = gtau[..., 0]
        gtau_beta = gtau[..., -1]
        gtau_mid = gtau[..., 1:-1]
        gtau_mid = 0.5 * (gtau_mid[..., ::2] + gtau_mid[..., 1::2])

        shape = list(gtau_mid.shape)
        shape[-1] += 2

        gtau = np.zeros(shape, dtype = complex)
        gtau[..., 0] = gtau_0
        gtau[..., -1] = gtau_beta
        gtau[..., 1:-1] = gtau_mid

        return gtau

    gtau = average_interior_bins(gtau)
    gtau_err = average_interior_bins(gtau_err)

    ### Reshape to rank 3
    
    assert( len(gtau.shape) == 5 )

    norbs, nspin, _, _, n_tau = gtau.shape
    shape = (norbs * nspin, norbs * nspin, n_tau)
    gtau, gtau_err = gtau.reshape(shape), gtau_err.reshape(shape)

    ### generate triqs Green function with same dimension than
    ### the input; this may not be a good idea if worm is used
    ### since then the output can have another structure...
    from triqs.gf import MeshImTime, BlockGf

    tau_mesh = MeshImTime(beta, 'Fermion', n_tau)

    G_tau_data = BlockGf(mesh=tau_mesh, gf_struct=gf_struct)
    G_tau_error = BlockGf(mesh=tau_mesh, gf_struct=gf_struct)

    gtau = exchange_fastest_running_index_ffw(gtau)

    ### read out blocks from full w2dyn matrices
    offset = 0
    for name, _ in G_tau_data:

        size1, size2 = G_tau_data[name].target_shape        
        assert( size1 == size2 )        
        size_block = size1

        gtau_block = gtau[offset:offset+size_block, offset:offset+size_block, :]
        gtau_err_block = gtau_err[offset:offset+size_block, offset:offset+size_block, :]

        G_tau_data[name].data[:] = -gtau_block.transpose(2, 0, 1)
        G_tau_error[name].data[:] = -gtau_err_block.transpose(2, 0, 1)
        
        offset += size_block

    return G_tau_data, G_tau_error

# ----------------------------------------------------------------------    
def triqs_gf_to_w2dyn_ndarray_g_wosos_beta_niw(G_iw):

    """ Convert a spin-block Triqs imaginary frequenca response function 
    (BlockGF) to  W2Dynamics ndarray format with indices [wosos]
    where iw is imaginary frequency, o is orbital, s is spin index. 

    Returns: 
    g_wosos : ndarray with the response function data
    beta : inverse temperature
    niw : number of imaginary frequency points

    Author: Hugo U. R. Strand (2019) """

    ### test gf object
    full_size = 0
    for name, g in G_iw:
        size_block = g.data.shape[-1]
        full_size += size_block

    assert full_size % 2 == 0, "Number of flavours cannot be odd in w2dyn_cthyb!"

    beta = G_iw.mesh.beta
    iw = np.array([ np.real(w) + 1.0j*np.imag(w) for w in G_iw.mesh ])
    niw = len(iw)
    np.testing.assert_almost_equal(np.imag(iw[niw//2]) * beta, np.pi)
    
    g_stoo = np.array([ G_iw.data for block_name, G_iw in G_iw ])
    nblocks, nt, size1, size2 = g_stoo.shape

    assert( size1 == size2 )

    ### the general back-conversion of the numpy arrays to triqs objects will
    ### anyway be ugly, therefore it does not matter for this 
    ### conversion either....

    g_tff = np.zeros(shape=(niw,full_size,full_size),dtype=G_iw.data.dtype)

    ### make one big blockdiagonal matrix 
    offset = 0
    for nb in range(nblocks):

        size_block = g_stoo[nb,:,:,:].shape[-1]
        g_tff[:,offset:offset+size_block,offset:offset+size_block] = g_stoo[nb,:,:,:]

        offset += size_block

    ### shape into spin structure
    g_tosos = g_tff.reshape(niw,full_size//2,2,full_size//2,2)

    return g_tosos, beta, niw

# ----------------------------------------------------------------------    
def w2dyn_ndarray_to_triqs_BlockGF_iw_beta_niw(giw, n_iw, beta, gf_struct):

    """ Convert a W2Dynamics ndarray format with indices [wosos] or [wff]
    where t is time, o is orbital, s is spin index, f is flavour
    to a block Triqs Matsubara response function 

    Takes: 
    giw  : Green function as numpy array
    beta : inverse temperature
    niw : number of Matsubaras
    gf_struct: desired block structure of triqs GF

    Returns: 
    BlockGF : triqs block Green function the response function data

    Author: Hugo U. R. Strand (2019) """

    ### check number of Matsubaras is correct and as last dimension
    ### the variable n_iw has only positive Matsubaras, the objects
    ### have both negative and positive
    n_iw_check = giw.shape[-1]/2
    assert n_iw_check == n_iw
    
    ### check if giw is 3 or 5 dimensional, and make it 3 dimensional
    if len(giw.shape) == 5:
        norbs = giw.shape[0]
        nspin = giw.shape[1]
        nflavour = norbs * nspin
        giw = giw.reshape(nflavour, nflavour, 2*n_iw)
    elif len(giw.shape) == 3:
        nflavour = giw.shape[0]
    else: 
       raise Exception("giw array must be 3 or 5 dimensional with iw as last dimension!")

    ### generate triqs Green function with same dimension than
    ### the input; this may not be a good idea if worm is used
    ### since then the output can have another structure...
    from triqs.gf import MeshImFreq, BlockGf
    iw_mesh = MeshImFreq(beta, 'Fermion', n_iw)
    G_iw = BlockGf(mesh=iw_mesh, gf_struct=gf_struct)

    ### read out blocks from full w2dyn matrices
    offset = 0
    for name, g_iw in G_iw:

        size1 = G_iw[name].data.shape[-1]
        size2 = G_iw[name].data.shape[-2]
        assert( size1 == size2 )
        size_block = size1
        giw_block = giw[offset:offset+size_block, offset:offset+size_block, :]
        g_iw.data[:] = giw_block.transpose(2, 0, 1)

        offset += size_block

    return G_iw


def w2dyn_g4iw_worm_to_triqs_block2gf(g4iw, beta, norb, gf_struct,
                                      qtype=(lambda x: x.mean())):
    """Converts a dictionary mapping zero-padded five digits long string
    representations of compound indices to components of the
    two-particle Green's function as ndarrays in DistributedSample objects with two fermionic
    frequency indices nu, nu' and one bosonic frequency index omega in
    the order [nu, nu', omega], as produced by w2dynamics, to a Triqs
    block Green's function on a Matsubara frequency mesh with one
    bosonic and two fermionic frequencies. Missing components are
    filled with zeros.

    Takes:
    g4iw : mapping from compound indices to components of the two-particle Green's function
    beta : inverse temperature
    norb : number of orbitals
    gf_struct : block structure of the triqs GF
    qtype : type / statistic of quantity to extract

    Returns:
    Block2Gf : requested statistic of the two-particle Green's function with one bosonic and two fermionic frequencies

    Author: Alexander Kowalski (2019) """
    from triqs.gf import Gf, Block2Gf, MeshImFreq, MeshProduct
    from w2dyn.auxiliaries import compound_index as ci

    # get number of positive freqs from a component of the result
    for i in range(100000):
        try:
            arr = qtype(g4iw["g4iw-worm/{:05}".format(i)])
            n4iwf, n4iwf_check, n4iwb = arr.shape
            assert(n4iwf == n4iwf_check)
            n4iwf, n4iwb = n4iwf//2, n4iwb//2 + 1
            break
        except KeyError:
            continue
        except AssertionError:
            raise ValueError("At least one component of g4iw-worm has an incorrect shape: should be (n4iwf, n4iwf, n4iwb)")

        return ValueError("g4iw-worm does not contain any valid components")

    # Generate frequency mesh with one bosonic frequency first and two
    # fermionic frequencies, which is the same format as used by triqs
    # cthyb
    iwmesh = MeshProduct(MeshImFreq(beta=beta, S='Boson', n_max=n4iwb),
                         MeshImFreq(beta=beta, S='Fermion', n_max=n4iwf),
                         MeshImFreq(beta=beta, S='Fermion', n_max=n4iwf))

    # Piece blocks for the triqs block Green's function together from
    # individual components, looping over blocks and then indices,
    # with offsets keeping track of the previous block sizes for
    # constructing the right w2dynamics compound indices
    offset1 = 0
    G4iw_blocks = []
    for name1, size1 in gf_struct:
        offset2 = 0
        subblocks = []
        G4iw_blocks.append(subblocks)
        for name2, size2 in gf_struct:
            G4iw_block = Gf(mesh=iwmesh, target_shape=[size1,
                                                       size1,
                                                       size2,
                                                       size2])
            subblocks.append(G4iw_block)

            for i in range(size1):
                for j in range(size1):
                    for k in range(size2):
                        for l in range(size2):
                            # we assume that spin is desired to be the
                            # slowest changing index in the triqs
                            # block structure, so we get orbital
                            # indices for the compound index from the
                            # block index by modulo and spin indices
                            # by integer division
                            cindex = ci.component2index_general(norb, 4,
                                    np.array([offset1 + i, offset1 + j,
                                              offset2 + k, offset2 + l]) % norb,
                                    np.array([offset1 + i, offset1 + j,
                                              offset2 + k, offset2 + l]) // norb)
                            try:
                                G4iw_block.data[:, :, :, i, j, k, l] = beta * qtype(g4iw[(
                                    "g4iw-worm/{:05}".format(cindex))]).transpose(2, 0, 1)
                            except KeyError:
                                G4iw_block.data[:, :, :, i, j, k, l] = 0.0
            offset2 += size2
        offset1 += size1

    G4iw_triqs = Block2Gf([x[0] for x in gf_struct],
                          [x[0] for x in gf_struct],
                          G4iw_blocks)

    return G4iw_triqs


def exchange_fastest_running_index_ff(array):

    assert len(array.shape) == 2, "length of array must be 2, but is %i" %len(array)

    t1 = array.shape[0]
    t2 = array.shape[1]

    assert t1 == t2
    nflav = t1

    array = array.reshape(nflav//2, 2, nflav//2, 2)
    array = array.transpose(1, 0, 3, 2)
    array = array.reshape(nflav, nflav)

    return array

def exchange_fastest_running_index_ffw(array):

    assert len(array.shape) == 3, "length of array must be 3, but is %i" %len(array)

    t1 = array.shape[0]
    t2 = array.shape[1]
    Niw = array.shape[2]

    assert t1 == t2
    nflav = t1

    array = array.reshape(nflav//2, 2, nflav//2, 2, Niw)
    array = array.transpose(1, 0, 3, 2, 4)
    array = array.reshape(nflav, nflav, Niw)

    return array

# ----------------------------------------------------------------------    
#if __name__ == '__main__':
def generate_testimpurity_with_triqs(norb, ntau, beta):

    #gf_struct, Delta_tau, H_loc = get_test_impurity_model(norb=2, ntau=1000, beta=10.0)
    gf_struct, Delta_tau, H_loc = get_test_impurity_model(norb, ntau, beta)

    # -- Convert the impurity model to ndarrays with W2Dynamics format
    # -- O : composite spin and orbital index
    # -- s, o : pure spin and orbital indices
    
    from pyed.OperatorUtils import fundamental_operators_from_gf_struct
    from pyed.OperatorUtils import quadratic_matrix_from_operator
    from pyed.OperatorUtils import quartic_tensor_from_operator

    fundamental_operators = fundamental_operators_from_gf_struct(gf_struct)
    #print "fundamental_operators ", fundamental_operators 
    
    t_OO = quadratic_matrix_from_operator(H_loc, fundamental_operators)
    U_OOOO = quartic_tensor_from_operator(H_loc, fundamental_operators, perm_sym=True)

    #print "t_OO", t_OO

    # -- Reshape tensors by breaking out the spin index separately
    
    t_osos = NO_to_Nos(t_OO, spin_first=True)
    U_osososos = NO_to_Nos(U_OOOO, spin_first=True)

    # -- Extract hybridization as ndarray from Triqs response function (BlockGf)
    
    delta_tosos, beta, ntau = triqs_gf_to_w2dyn_ndarray_g_tosos_beta_ntau(Delta_tau)

    #print 'beta =', beta
    #print 'ntau =', ntau
    #print 't_osos.shape =', t_osos.shape
    #print 'U_osososos.shape =', U_osososos.shape
    #print 'delta_tosos.shape =', delta_tosos.shape

    return norb, beta, ntau, t_osos, U_OOOO, delta_tosos
