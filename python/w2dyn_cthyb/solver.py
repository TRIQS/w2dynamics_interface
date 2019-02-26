""" 
W2Dynamics wrapper for the Triqs library

Authors: Andreas Hausoel, Hugo U. R. Strand, Nils Wentzell (2019)

"""
import os, sys
import tempfile
import numpy as np
from scipy.linalg import block_diag

import pytriqs.utility.mpi as mpi
from pytriqs.gf import Fourier
from pytriqs.gf import MeshImTime, MeshImFreq, BlockGf
from pytriqs.gf.tools import conjugate

import w2dyn.auxiliaries.CTQMC
import w2dyn.dmft.impurity as impurity
import w2dyn.auxiliaries.config as config

from pyed.OperatorUtils import fundamental_operators_from_gf_struct
from pyed.OperatorUtils import quadratic_matrix_from_operator
from pyed.OperatorUtils import quartic_tensor_from_operator

from converters import NO_to_Nos
from converters import w2dyn_ndarray_to_triqs_BlockGF_tau_beta_ntau
from converters import w2dyn_ndarray_to_triqs_BlockGF_iw_beta_niw
from converters import triqs_gf_to_w2dyn_ndarray_g_tosos_beta_ntau
from extractor import extract_deltaiw_and_tij_from_G0

class Solver():
    
    def __init__(self, beta, gf_struct, n_iw=1025, n_tau=10001, n_l=30, delta_interface=False):
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
        self.gf_struct= gf_struct
        self.n_iw = n_iw
        self.n_tau = n_tau
        self.n_l = n_l
        self.delta_interface = delta_interface

        self.tau_mesh = MeshImTime(beta, 'Fermion', n_tau)
        self.iw_mesh = MeshImFreq(beta, 'Fermion', n_iw)

        if self.delta_interface:
            self.Delta_tau = BlockGf(mesh=self.tau_mesh, gf_struct=self.gf_struct)
        else:
            self.G0_iw = BlockGf(mesh=self.iw_mesh, gf_struct=gf_struct)

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
        n_warmup_cycles = params_kw.pop("n_warmup_cycles", 100000) ### default
        length_cycle = params_kw.pop("length_cycle", 50)
        h_int = params_kw.pop("h_int")
        self.last_solve_params = { "n_cycles": n_cycles, "n_warmup_cycles": n_warmup_cycles,
            "length_cycle": length_cycle, "h_int": h_int }
        
        if self.delta_interface:
            h_0 = params_kw.pop("h_0")
            self.last_solve_params["h_0"] = h_0

        if isinstance(self.gf_struct,dict):
            print "WARNING: gf_struct should be a list of pairs [ [str,[int,...]], ...], not a dict"
            self.gf_struct = [ [k, v] for k, v in self.gf_struct.iteritems() ]

        fundamental_operators = fundamental_operators_from_gf_struct(self.gf_struct)

        ### Andi: the definition in the U-Matrix in w2dyn is
        ### 1/2 \sum_{ijkl} U_{ijkl} cdag_i cdag_j c_l c_k
        ###                                         !   !
        ### a factor of 2 is needed to compensate the 1/2, and a minus for 
        ### exchange of the annihilators; is this correct for any two particle interaction term?
        U_ijkl = -2.0 * quartic_tensor_from_operator(
                h_int, fundamental_operators, perm_sym=False)
        U_ijkl = U_ijkl.reshape(norb, 2, norb, 2, norb, 2, norb, 2)
        U_ijkl = U_ijkl.transpose(1,0, 3,2, 5,4, 7,6)
        U_ijkl = U_ijkl.reshape(norb*2, norb*2, norb*2, norb*2)

        if self.delta_interface:
            ### TODO: check that h_0 only contains quadratic terms and is gf_struct compatible
            t_ij_matrix = quadratic_matrix_from_operator(h_0, fundamental_operators)
        else:
        ### TODO: check that H_int only contains quartic terms and is gf_struct compatible
            Delta_iw, t_ij_lst = extract_deltaiw_and_tij_from_G0(self.G0_iw, self.gf_struct)
            Delta_iw = conjugate(Delta_iw) # in w2dyn Delta is a hole propagator

            self.Delta_tau = BlockGf(mesh=self.tau_mesh, gf_struct=self.gf_struct)
            self.Delta_tau << Fourier(Delta_iw)

            assert len(t_ij_lst) in set([1, 2, 4]), \
                  "For now t_ij_lst must not contain more than 4 blocks; generalize it!"
            t_ij_matrix = block_diag(*t_ij_lst)

        t_ij_matrix *= -1 # W2Dynamics sign convention

        ### transform t_ij from (f,f) to (o,s,o,s) format
        t_osos_tensor = NO_to_Nos(t_ij_matrix, spin_first=True)
        norb = t_osos_tensor.shape[0]

        ### TODO: triqs solver takes G0 and converts it into F(iw) and F(tau)
        ### but we directly need F(tau)

        ftau, _, __ = triqs_gf_to_w2dyn_ndarray_g_tosos_beta_ntau(self.Delta_tau)
        print 'ftau.shape', ftau.shape

        ftau = ftau.transpose(0,2,1,4,3)
        # print "ftau", ftau

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

        cfg_file = tempfile.NamedTemporaryFile(delete=False)
        cfg_file.write(Parameters_in)
        cfg_file.close()
        
        ### read w2dyn parameter file; later we will replace this by a 
        ### converter of triqs-parameters to w2dyn-parameters

        key_value_args={}
        cfg =  config.get_cfg(cfg_file.name, key_value_args, err=sys.stderr)
        cfg["QMC"]["offdiag"] = 1

        ### in case of the complex
        #cfg["QMC"]["complex"] = 1
        #cfg["QMC"]["use_phase"] = 1
        cfg["QMC"]["Percentage4OperatorMove"] = 0.005
        cfg["QMC"]["PercentageGlobalMove"] = 0.005

        os.remove(cfg_file.name) # remove temp file with input parameters

        ### I now write the triqs parameters into the cfg file;
        ### we may later do this with dictionaries
        ### in a more sophisticated way

        cfg["General"]["beta"] = self.beta
        cfg["QMC"]["Niw"] = self.n_iw
        cfg["QMC"]["Ntau"] = self.n_tau * 2 # use double resolution bins & down sample to Triqs l8r
        cfg["QMC"]["NLegMax"] = self.n_l
        cfg["QMC"]["NLegOrder"] = self.n_l

        cfg["QMC"]["Nwarmups"] = length_cycle * n_warmup_cycles
        cfg["QMC"]["Nmeas"] = n_cycles
        cfg["QMC"]["Ncorr"] = length_cycle

        #cfg["QMC"]["statesampling"] = 1
        #for name in cfg["QMC"]:
            #print name, " = ", cfg["QMC"][name]
        # cfg["General"]["FFType"] = "plain-full"

        #print "cfg", cfg

        ### initialize the solver; it needs the config-string
        Nseed = 1 + mpi.rank
        use_mpi = False
        mpi_comm = mpi.world
        solver = impurity.CtHybSolver(cfg, Nseed, 0,0,0, False, mpi_comm)

        ### generate dummy input that we don't necessarily need
        niw     = 2*cfg["QMC"]["Niw"]
        g0inviw = np.zeros(shape=(2*self.n_iw, norb, 2, norb, 2))
        fiw     = np.zeros(shape=(2*self.n_iw, norb, 2, norb, 2))
        fmom    = np.zeros(shape=(2, norb, 2, norb, 2))
        symmetry_moves = ()
        paramag = False
        atom = config.atomlist_from_cfg(cfg, norb)[0]

        ### we begin with real not complex calculations
        g0inviw = np.real(g0inviw)
        fiw = np.real(fiw)
        fmom = np.real(fmom)
        ftau = np.real(ftau)
        muimp = np.real(t_osos_tensor)
        U_ijkl = np.real(U_ijkl)
 
        ### this is for complex
        #muimp = t_osos

        print "ftau.shape", ftau.shape

        ### here the properties of the impurity will be defined
        ### impurity problem for real code:
        imp_problem = impurity.ImpurityProblem(
            self.beta, g0inviw, fiw, fmom, ftau,
            muimp, atom.dd_int, None, None, symmetry_moves,
            paramag)
        #### impurity problem for complex code:
        #imp_problem = impurity.ImpurityProblem(
            #self.beta, g0inviw, fiw, fmom, ftau,
            #muimp, muimp, atom.dd_int, None, None, symmetry_moves,
            #paramag)

        print "\n" + "."*40

        ### hardcode the set of conserved quantities to number of electrons
        ### and activate the automatic minimalisation procedure of blocks 
        ### ( QN "All" does this)
        imp_problem.interaction.quantum_numbers = ( "Nt", "All" )
        #imp_problem.interaction.quantum_numbers = ( "Nt", "Szt", "Qzt" ) 
        #imp_problem.interaction.quantum_numbers = ( "Nt", "Szt" )
        #imp_problem.interaction.quantum_numbers = ( "Nt" )

        ### feed impurity problem into solver
        solver.set_problem(imp_problem)

        ### overwrite dummy umatrix in solver class
        #print "solver.umatrix.shape", solver.umatrix.shape
        #print "U_ijkl.shape", U_ijkl.shape
        solver.umatrix = U_ijkl

        ### solve impurity problem 
        mccfgcontainer = []
        iter_no = 1
        ### for real
        result = solver.solve(iter_no, mccfgcontainer)
        ### for complex
        #result = solver.solve(mccfgcontainer)
 
        gtau = result.other["gtau-full"]

        ### here comes the function for conversion w2dyn --> triqs
        self.G_tau, self.G_tau_error = w2dyn_ndarray_to_triqs_BlockGF_tau_beta_ntau(
            gtau, self.beta, self.gf_struct)

        self.G_iw = BlockGf(mesh=self.iw_mesh, gf_struct=self.gf_struct)

        ### I will use the FFT from triqs here...
        for name, g in self.G_tau:
            bl_size = g.target_shape[0]
            known_moments = np.zeros((4, bl_size, bl_size), dtype=np.complex)
            for i in range(bl_size):
                known_moments[1,i,i] = 1

            self.G_iw[name].set_from_fourier(g, known_moments)
