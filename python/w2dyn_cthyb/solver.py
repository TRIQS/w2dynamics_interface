""" 
W2Dynamics wrapper for the Triqs library

Authors: Andreas Hausoel, Hugo U. R. Strand, Nils Wentzell (2019)

"""
import os, sys
import tempfile
import numpy as np
from scipy.linalg import block_diag

import triqs.utility.mpi as mpi

from triqs.gf import Fourier
from triqs.gf import MeshImTime, MeshImFreq, BlockGf
from triqs.gf.tools import conjugate
from triqs.gf.block_gf import fix_gf_struct_type
from triqs.operators.util.extractors import *

import w2dyn.auxiliaries.CTQMC
import w2dyn.dmft.impurity as impurity
import w2dyn.auxiliaries.config as config
import w2dyn.auxiliaries.statistics as stat

from .converters import NO_to_Nos
from .converters import w2dyn_ndarray_to_triqs_BlockGF_tau_beta_ntau
from .converters import w2dyn_ndarray_to_triqs_BlockGF_iw_beta_niw
from .converters import triqs_gf_to_w2dyn_ndarray_g_tosos_beta_ntau
from .converters import w2dyn_g4iw_worm_to_triqs_block2gf
from .extractor import extract_deltaiw_and_tij_from_G0

class Solver():
    
    def __init__(self, beta, gf_struct, n_iw=1025, n_tau=10001, n_l=30, delta_interface=False, complex=False):
        """Constructor setting up response function parameters

        Arguments:
        beta : inverse temperature
        gf_struct : Triqs Green's function block structure
        n_iw : number of Matsubara frequencies
        n_tau : number of imaginary time points
        """

        self.constr_params = { "beta": beta, "gf_struct": gf_struct, "n_iw": n_iw,
                "n_tau": n_tau, "n_l": n_l, "delta_interface": delta_interface }

        self.beta = beta
        self.gf_struct = fix_gf_struct_type(gf_struct)
        self.n_iw = n_iw
        self.n_tau = n_tau
        self.n_l = n_l
        self.delta_interface = delta_interface
        self.complex = complex

        self.tau_mesh = MeshImTime(beta, 'Fermion', n_tau)
        self.iw_mesh = MeshImFreq(beta, 'Fermion', n_iw)

        if self.delta_interface:
            self.Delta_tau = BlockGf(mesh=self.tau_mesh, gf_struct=self.gf_struct)
        else:
            self.G0_iw = BlockGf(mesh=self.iw_mesh, gf_struct=self.gf_struct)

    def solve(self, **params_kw):
        """Solve impurity model 

        Arguments:
        n_cycles : number of Monte Carlo cycles
        n_warmup_cycles : number of warmub Monte Carlo cycles
        length_cycle : number of proposed moves per cycle
        h_int : interaction Hamiltonian as a quartic triqs operator
        h_0 : quadratic part of the local Hamiltonian
              (only required if delta_interface=true has been specified during construction)
        """

        n_cycles = params_kw.pop("n_cycles")  ### what does the True or False mean?
        n_warmup_cycles = params_kw.pop("n_warmup_cycles", 5000) ### default
        max_time = params_kw.pop("max_time", -1)
        worm = params_kw.pop("worm", False)
        percentageworminsert = params_kw.pop("PercentageWormInsert", 0.20)
        percentagewormreplace = params_kw.pop("PercentageWormReplace", 0.20)
        wormcomponents = params_kw.pop("worm_components", None)

        length_cycle = params_kw.pop("length_cycle", 50)
        h_int = params_kw.pop("h_int")
        self.last_solve_params = { "n_cycles": n_cycles, "n_warmup_cycles": n_warmup_cycles,
            "length_cycle": length_cycle, "h_int": h_int }
        
        if self.delta_interface:
            h_0 = params_kw.pop("h_0")
            self.last_solve_params["h_0"] = h_0

        random_seed = params_kw.pop("random_seed", 1)
        move_double = params_kw.pop("move_double", True)
        measure_G_l = params_kw.pop("measure_G_l", True)
        measure_G_tau = params_kw.pop("measure_G_tau", True)
        measure_pert_order = params_kw.pop("measure_pert_order", False)
        statesampling = params_kw.pop("statesampling", False)
        flavourchange_moves = params_kw.pop("flavourchange_moves", False)
        move_global_prob = params_kw.pop("flavomove_global_prob", 0.005)
        measure_g4iw_ph = params_kw.pop("measure_G2_iw_ph", False)
        n4iwf = params_kw.pop("measure_G2_n_fermionic", 30)
        n4iwb = params_kw.pop("measure_G2_n_bosonic", 30) - 1

        ### Andi: the definition in the U-Matrix in w2dyn is
        ### 1/2 \sum_{ijkl} U_{ijkl} cdag_i cdag_j c_l c_k
        ###                                         !   !
        ### a factor of 2 is needed to compensate the 1/2, and a minus for 
        ### exchange of the annihilators; is this correct for any two particle interaction term?

        U_ijkl = dict_to_matrix(extract_U_dict4(h_int), self.gf_struct)

        ### Make sure that the spin index is the fastest running variable
        norb = U_ijkl.shape[0]//2
        U_ijkl = U_ijkl.reshape(2,norb, 2,norb, 2,norb, 2,norb)
        U_ijkl = U_ijkl.transpose(1,0, 3,2, 5,4, 7,6)
        U_ijkl = U_ijkl.reshape(norb*2, norb*2, norb*2, norb*2)

        if self.delta_interface:
            t_ij_matrix = dict_to_matrix(extract_h_dict(h_0), self.gf_struct)
        else:
            Delta_iw, t_ij_lst = extract_deltaiw_and_tij_from_G0(self.G0_iw, self.gf_struct)

            self.Delta_tau = BlockGf(mesh=self.tau_mesh, gf_struct=self.gf_struct)
            self.Delta_tau << Fourier(Delta_iw)

            assert len(t_ij_lst) in set([1, 2, 4]), \
                  "For now t_ij_lst must not contain more than 4 blocks; generalize it!"
            t_ij_matrix = block_diag(*t_ij_lst)

        # in w2dyn Delta is a hole propagator
        for bl, Delta_bl in self.Delta_tau:
            Delta_bl.data[:] = -Delta_bl.data[::-1,...]

        t_ij_matrix *= -1 # W2Dynamics sign convention

        ### transform t_ij from (f,f) to (o,s,o,s) format

        norb = int(t_ij_matrix.shape[0]/2)
        t_ij_matrix = t_ij_matrix.reshape(2, norb, 2, norb)
        t_osos_tensor = t_ij_matrix.transpose(1,0, 3,2)

        ftau, _, __ = triqs_gf_to_w2dyn_ndarray_g_tosos_beta_ntau(self.Delta_tau)

        ### now comes w2dyn!
        # Make a temporary files with input parameters
        
        Parameters_in = """#asdf
[General]
[Atoms]
[[1]]
Nd = %i
Hamiltonian = Kanamori
[QMC]
TaudiffMax = -1.0""" % norb

        cfg_file = tempfile.NamedTemporaryFile(delete=False, mode = "w")
        cfg_file.write(Parameters_in)
        cfg_file.close()
        
        ### read w2dyn parameter file; later we will replace this by a 
        ### converter of triqs-parameters to w2dyn-parameters

        key_value_args={}
        cfg =  config.get_cfg(cfg_file.name, key_value_args, err=sys.stderr)

        ### check if Delta_tau is diagonal matrix, and set w2dyn parameter
        ### offdiag accordingly
        max_blocksize = 0
        for name, d in self.Delta_tau:

            blocksize = d.data.shape[-1]
            #print "blocksize", blocksize
            if blocksize > max_blocksize:
                max_blocksize = blocksize

        if max_blocksize == 1:
            cfg["QMC"]["offdiag"] = 0
        else:
            cfg["QMC"]["offdiag"] = 1

        ### complex worms are not yet existing
        if self.complex and worm:
            print('complex and worm together not yet implemented')
            exit()

        if self.complex:
            cfg["QMC"]["complex"] = 1
            cfg["QMC"]["use_phase"] = 1

            ### check if offdiag is set; complex makes no sense without offdiag
            assert cfg["QMC"]["offdiag"] != 0, \
                  "Complex does not make sense for diagonal Delta_tau!"


        if move_double:
            cfg["QMC"]["Percentage4OperatorMove"] = 0.005
        else:
            cfg["QMC"]["Percentage4OperatorMove"] = 0

        cfg["QMC"]["PercentageGlobalMove"] = move_global_prob

        if flavourchange_moves:
            cfg["QMC"]["flavourchange_moves"] = 1
        else:
            cfg["QMC"]["flavourchange_moves"] = 0

        os.remove(cfg_file.name) # remove temp file with input parameters

        ### I now write the triqs parameters into the cfg file;
        ### we may later do this with dictionaries
        ### in a more sophisticated way

        cfg["General"]["beta"] = self.beta
        cfg["QMC"]["Niw"] = self.n_iw
        if measure_G_tau:
            cfg["QMC"]["Ntau"] = self.n_tau * 2 # use double resolution bins & down sample to Triqs l8r
        else:
            cfg["QMC"]["Ntau"] = 1
        if measure_G_l:
            cfg["QMC"]["NLegMax"] = self.n_l
            cfg["QMC"]["NLegOrder"] = self.n_l
        else:
            cfg["QMC"]["NLegMax"] = 1
            cfg["QMC"]["NLegOrder"] = 1

        cfg["QMC"]["Nwarmups"] = length_cycle * n_warmup_cycles
        cfg["QMC"]["Nmeas"] = n_cycles
        cfg["QMC"]["measurement_time"] = max_time
        cfg["QMC"]["Ncorr"] = length_cycle

        if statesampling:
            cfg["QMC"]["statesampling"] = 1
        else:
            cfg["QMC"]["statesampling"] = 0

        if worm:
            cfg["QMC"]["WormMeasGiw"] = 1
            cfg["QMC"]["WormMeasGtau"] = 1
            cfg["QMC"]["WormSearchEta"] = 1

            ### set worm parameters to some default values if not set by user
            if percentageworminsert != 0.0:
                cfg["QMC"]["PercentageWormInsert"] = percentageworminsert
            else:
                cfg["QMC"]["PercentageWormInsert"] = 0.2
            if percentagewormreplace != 0.0:
                cfg["QMC"]["PercentageWormReplace"] = percentagewormreplace
            else:
                cfg["QMC"]["PercentageWormReplace"] = 0.2

        if mpi.rank == 0:
            print(' ')
            print('specifications for w2dyn:')
            print('cfg["QMC"]["offdiag"] ',  cfg["QMC"]["offdiag"])
            print('cfg["QMC"]["Percentage4OperatorMove"] ',  cfg["QMC"]["Percentage4OperatorMove"])
            print('cfg["QMC"]["flavourchange_moves"] ',  cfg["QMC"]["flavourchange_moves"])
            print('cfg["QMC"]["statesampling"] ', cfg["QMC"]["statesampling"])


        ### initialize the solver; it needs the config-string
        Nseed = random_seed + mpi.rank
        use_mpi = False
        import mpi4py
        mpi_comm = mpi4py.MPI.COMM_WORLD
        solver = impurity.CtHybSolver(cfg, Nseed, 0,0,0, False, mpi_comm)

        ### generate dummy input that we don't necessarily need
        niw     = 2*cfg["QMC"]["Niw"]
        g0inviw = np.zeros(shape=(2*self.n_iw, norb, 2, norb, 2))
        fiw     = np.zeros(shape=(2*self.n_iw, norb, 2, norb, 2))
        fmom    = np.zeros(shape=(2, norb, 2, norb, 2))
        symmetry_moves = ()
        paramag = False
        atom = config.atomlist_from_cfg(cfg, norb)[0]

        ### if calculation not complex, the solver must have only
        ### real arrays as input
        if self.complex:
            muimp = t_osos_tensor
        else:
            g0inviw = np.real(g0inviw)
            fiw = np.real(fiw)
            fmom = np.real(fmom)
            ftau = np.real(ftau)
            muimp = np.real(t_osos_tensor)
            U_ijkl = np.real(U_ijkl)
 
        ### here the properties of the impurity will be defined
        imp_problem = impurity.ImpurityProblem(
            self.beta, g0inviw, fiw, fmom, ftau,
            muimp, atom.dd_int, None, None, symmetry_moves,
            paramag)

        print("\n" + "."*40)

        ### hardcode the set of conserved quantities to number of electrons
        ### and activate the automatic minimalisation procedure of blocks 
        ### ( QN "All" does this)
        #print "imp_problem.interaction.quantum_numbers ",  imp_problem.interaction.quantum_numbers
        imp_problem.interaction.quantum_numbers = ( "Nt", "All" )
        #imp_problem.interaction.quantum_numbers = ( "Nt", "Szt", "Qzt" ) 
        #imp_problem.interaction.quantum_numbers = ( "Nt", "Szt" )
        #imp_problem.interaction.quantum_numbers = ( "Nt" )

        ### feed impurity problem into solver
        solver.set_problem(imp_problem)

        ### solve impurity problem
        mccfgcontainer = []
        iter_no = 1
        if measure_G_tau or measure_G_l or measure_pert_order:
            if self.complex:
                solver.set_problem(imp_problem)
                solver.umatrix = U_ijkl
                result = solver.solve(mccfgcontainer)
                gtau = result.other["gtau-full"]
            else:
                if not worm:
                    solver.set_problem(imp_problem)
                    solver.umatrix = U_ijkl
                    result = solver.solve(iter_no, mccfgcontainer)
                    gtau = result.other["gtau-full"]
                else:

                  gtau = np.zeros(shape=(1, norb, 2, norb, 2, 2*self.n_tau))

                  from w2dyn.auxiliaries.compound_index import index2component_general

                  components = []

                  for comp_ind in range(1,(2*norb)**2+1):

                      tmp = index2component_general(norb, 2, int(comp_ind))

                      ### check if ftau is nonzero

                      bands = tmp[1]
                      spins = tmp[2]

                      b1 = bands[0]
                      b2 = bands[1]
                      s1 = spins[0]
                      s2 = spins[1]

                      all_zeros = not np.any(ftau[:,b1,s1,b2,s2]>1e-5)

                      if not all_zeros:
                          components = np.append(components, comp_ind)

                  if mpi.rank == 0:
                      print('worm components to measure: ', components)

                  ### divide either max_time Nmeas among the nonzero components
                  if max_time <= 0:
                      cfg["QMC"]["Nmeas"] = int(cfg["QMC"]["Nmeas"] / float(len(components)))
                  else:
                      cfg["QMC"]["measurement_time"] = int(float(max_time) / float(len(components)))

                  for comp_ind in components:

                      if mpi.rank == 0:
                          print('--> comp_ind', comp_ind)

                      solver.set_problem(imp_problem)
                      solver.umatrix = U_ijkl
                      result_aux, result = solver.solve_component(1,2,comp_ind,mccfgcontainer)

                      for i in list(result.other.keys()):

                          if "gtau-worm" in i:
                              gtau_name = i

                      tmp = index2component_general(norb, 2, int(comp_ind))

                      ### check if ftau is nonzero

                      bands = tmp[1]
                      spins = tmp[2]

                      b1 = bands[0]
                      b2 = bands[1]
                      s1 = spins[0]
                      s2 = spins[1]

                      # Remove axis 0 from local samples by averaging, so
                      # no data remains unused even if there is more than
                      # one local sample (should not happen)
                      gtau[0, b1, s1, b2, s2, :] = np.mean(result.other[gtau_name],
                                                           axis=0)
                  gtau = stat.DistributedSample(gtau, mpi_comm, ntotal=mpi.size)

        if measure_g4iw_ph:
            # Set necessary configuration parameters for the solver
            # and construct a new one so the Fortran module also gets
            # them (config is currently passed in the constructor)
            cfg["QMC"]["FourPnt"] = 8
            cfg["QMC"]["N4iwf"] = n4iwf
            cfg["QMC"]["N4iwb"] = n4iwb
            cfg["QMC"]["PercentageWormInsert"] = percentageworminsert
            cfg["QMC"]["PercentageWormReplace"] = percentagewormreplace
            cfg["QMC"]["WormMeasG4iw"] = 1
            cfg["QMC"]["WormSearchEta"] = 1
            solver = impurity.CtHybSolver(cfg, Nseed, 0,0,0, False, mpi_comm)

            g4iw = dict()
            # the number of components could be reduced by restricting
            # it to the requested blocks (measure_G2_blocks solve
            # parameter) and the nonzero elements for the given
            # interaction (if easily determinable)
            mccfgcontainer = []
            for icomponent in (wormcomponents
                               if wormcomponents is not None
                               else range(1, (2*norb)**4+1)):

                if mpi.rank == 0:
                    print('Sampling worm component {}'.format(icomponent))

                solver.set_problem(imp_problem, cfg["QMC"]["FourPnt"])
                solver.umatrix = U_ijkl
                res_gen, res_g4comp = solver.solve_comp_stats(1, 4,
                                                              icomponent,
                                                              mccfgcontainer)

                # Collect components into right format for the
                # conversion function:
                # g4iw["g4iw-worm/{:05}".format(compound_index)]["value"]
                # should give the value of the component specified by
                # compound_index and replacing "value" by "error" the
                # error
                for i in res_g4comp.other.keys():
                    if "g4iw-worm" in i:
                        g4iw[i] = res_g4comp.other[i]

            self.G2_iw_ph = w2dyn_g4iw_worm_to_triqs_block2gf(g4iw,
                                                              self.beta,
                                                              norb,
                                                              self.gf_struct)
            self.G2_iw_ph_error = (
                w2dyn_g4iw_worm_to_triqs_block2gf(g4iw,
                                                  self.beta,
                                                  norb,
                                                  self.gf_struct,
                                                  qtype=(lambda x: x.stderr())))


        def bs_diagflat(bs_array):
            """Return an array with a shape extended compared to
            that of the argument by doubling the first two axes and
            fill it such that the returned array is diagonal with
            respect to both pairs (axes 1 and 3 and axes 2 and 4).
            """
            shape_bsonly = bs_array.shape[0:2]
            bsbs_shape = shape_bsonly + bs_array.shape
            bsbs_array = np.zeros(bsbs_shape, dtype=bs_array.dtype)
            for b in range(bs_array.shape[0]):
                for s in range(bs_array.shape[1]):
                    bsbs_array[b, s, b, s, ...] = bs_array[b, s, ...]
            return bsbs_array


        ### here comes the function for conversion w2dyn --> triqs
        if measure_G_tau:
            if cfg["QMC"]["offdiag"] == 0 and worm == 0:
                gtau = result.other["gtau"].apply(bs_diagflat)

            self.G_tau, self.G_tau_error = w2dyn_ndarray_to_triqs_BlockGF_tau_beta_ntau(
                gtau, self.beta, self.gf_struct)

            self.G_iw = BlockGf(mesh=self.iw_mesh, gf_struct=self.gf_struct)

            ### I will use the FFT from triqs here...
            for name, g in self.G_tau:
                bl_size = g.target_shape[0]
                known_moments = np.zeros((4, bl_size, bl_size), dtype=complex)
                for i in range(bl_size):
                    known_moments[1,i,i] = 1

                self.G_iw[name].set_from_fourier(g, known_moments)

        ### add perturbation order as observable
        #print 'measure_pert_order ', measure_pert_order 
        if measure_pert_order:
            hist = result.other["hist"]
            #print 'hist.shape', hist.shape

        ### GF in Legendre expansion
        if measure_G_l:
            Gl = result.other["gleg-full"]
            #print 'Gl.shape', Gl.shape
