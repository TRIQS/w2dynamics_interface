""" 

W2Dynamics wrapper for the Triqs library

Authors: Andreas Hausoel, Hugo U. R. Strand (2019)

"""
import os, sys
import tempfile
import numpy as np

from pytriqs.gf import MeshImTime,MeshImFreq, BlockGf
import pytriqs.utility.mpi as mpi

import w2dyn.auxiliaries.CTQMC

### here come the necessary imports form w2dyn dmft loop
import w2dyn.dmft.impurity as impurity
import w2dyn.auxiliaries.config as config

class Solver():
    
    def __init__(self, beta, gf_struct, n_iw=1025, n_tau=10001, n_l=30):

        self.beta = beta
        self.gf_struct= gf_struct
        self.n_iw = n_iw
        self.n_tau = n_tau
        self.n_l = n_l

        self.tau_mesh = MeshImTime(beta, 'Fermion', n_tau)
        self.Delta_tau_directly_passed = BlockGf(mesh=self.tau_mesh, gf_struct=gf_struct)

        self.tau_mesh = MeshImTime(self.beta, 'Fermion', self.n_tau)
        self.Delta_tau= BlockGf(mesh=self.tau_mesh, gf_struct=self.gf_struct)

        self.iw_mesh = MeshImFreq(beta, 'Fermion', n_iw)
        self.G0_iw = BlockGf(mesh=self.iw_mesh, gf_struct=gf_struct)

        self.G_iw = BlockGf(mesh=self.iw_mesh, gf_struct=gf_struct)

    def solve(self, **params_kw):

        self.n_cycles = params_kw.pop("n_cycles")  ### what does the True or False mean?
        self.n_warmup_cycles = params_kw.pop("n_warmup_cycles", 100000) ### default
        self.length_cycle = params_kw.pop("length_cycle", 50)
        self.h_int = params_kw.pop("h_int")
        
        #### load stuff from pyed
        from pyed.OperatorUtils import fundamental_operators_from_gf_struct
        from pyed.OperatorUtils import quadratic_matrix_from_operator
        from pyed.OperatorUtils import quartic_tensor_from_operator

        if isinstance(self.gf_struct,dict):
            print "WARNING: gf_struct should be a list of pairs [ [str,[int,...]], ...], not a dict"
            self.gf_struct = [ [k, v] for k, v in self.gf_struct.iteritems() ]


        ### I now also generate the fundamental operators out of gf_struct and save them
        from pyed.OperatorUtils import fundamental_operators_from_gf_struct
        fundamental_operators = fundamental_operators_from_gf_struct(self.gf_struct)
        #print "fundamental_operators ", fundamental_operators 

        ### extract t_ij and U_ijkl from gf_struct
        #print "extract t_ij and U_ijkl from gf_struct... "
        t_OO = quadratic_matrix_from_operator(self.h_int, fundamental_operators)

        ### sign convention, t_OO gets a minus sign
        t_OO = -t_OO

        ### use the extractor to get Delta(iw) and t_OO from the given G0
        from extractor import extract_deltaiw_and_tij_from_G0
        from pytriqs.gf import Fourier
        Delta_iw, t_OO_extr_list = extract_deltaiw_and_tij_from_G0(self.G0_iw, self.gf_struct)

        ### in w2dyn we need this for holes, therefore conjugate
        from pytriqs.gf.tools import conjugate
        Delta_iw = conjugate(Delta_iw)

        self.Delta_tau << Fourier(Delta_iw)

        print "compare delta_tau:"
        for block, dt in self.Delta_tau:
            d1 = self.Delta_tau[block].data
            #d2 = self.Delta_tau_directly_passed[block].data
            #np.testing.assert_array_almost_equal(d1,d2)

        from scipy.linalg import block_diag
        ### merge blocks into one big matrix
        print "len(t_OO_extr_list) ",  len(t_OO_extr_list)
        if len(t_OO_extr_list) == 1:
           t_OO_extr = block_diag(t_OO_extr_list[0])
        elif len(t_OO_extr_list) == 2:
           t_OO_extr = block_diag(t_OO_extr_list[0],t_OO_extr_list[1])
        elif len(t_OO_extr_list) == 4:
           t_OO_extr = block_diag(t_OO_extr_list[0],t_OO_extr_list[1],t_OO_extr_list[2],t_OO_extr_list[3])
        else:
           raise Exception("For now t_OO must not contain more than 4 blocks; generalize it!")

        #np.testing.assert_array_almost_equal(t_OO,t_OO_extr)

        ### sign convention, t_OO gets a minus sign
        t_OO_extr = -t_OO_extr

        ### we want to use the one extracted
        t_OO = t_OO_extr

        ### Andi: the definition in the U-Matrix in w2dyn is
        ### 1/2 \sum_{ijkl} U_{ijkl} cdag_i cdag_j c_l c_k
        ###                                         !   !
        ### a factor of 2 is needed to compensate the 1/2, and a minus for 
        ### exchange of the annihilators; is this correct for any two particle interaction term?
        U_OOOO = quartic_tensor_from_operator(self.h_int, fundamental_operators, perm_sym=True)
        U_OOOO *= -2.0

        ### the U tensor is not correct!
        #print "U_OOOO"
        #for i in range(0,2):
            #for j in range(0,2):
                #for k in range(0,2):
                    #for l in range(0,2):
                        #print i,j,k,l, U_OOOO[i,j,k,l]
        
        #exit()

        ### transform t_ij from (f,f) to (o,s,o,s) format
        from converters import NO_to_Nos
        t_osos = NO_to_Nos(t_OO, spin_first=True)
        #t_osos *= -1.0
        #print "t_osos", t_osos
        #print "t_osos.shape", t_osos.shape
        norb = t_osos.shape[0]


        ### TODO: triqs solver takes G0 and converts it into F(iw) and F(tau)
        ### but we directly need F(tau)

        from converters import triqs_gf_to_w2dyn_ndarray_g_tosos_beta_ntau

        ftau, _, __ = triqs_gf_to_w2dyn_ndarray_g_tosos_beta_ntau(self.Delta_tau)

        #print "ftau.shape", ftau.shape
        #print "ftau", ftau

        ### now comes w2dyn!
        import w2dyn.dmft.impurity as impurity
        import w2dyn.auxiliaries.config as config

        # Make a temporary files with input parameters
        
        Parameters_in = """#asdf
[General]
[Atoms]
[[1]]
Nd = %i
Hamiltonian = Kanamori
[QMC]
TaudiffMax = 2.0""" % norb

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

        ### I now write the triqs parameters into the cfg file; we may later do this with dictionaries
        ### in a more sophisticated way

        cfg["General"]["beta"] = self.beta
        cfg["QMC"]["Niw"] = self.n_iw
        cfg["QMC"]["Ntau"] = self.n_tau
        cfg["QMC"]["NLegMax"] = self.n_l
        cfg["QMC"]["NLegOrder"] = self.n_l

        cfg["QMC"]["Nwarmups"] = self.length_cycle * self.n_warmup_cycles
        cfg["QMC"]["Nmeas"] = self.n_cycles
        cfg["QMC"]["Ncorr"] = self.length_cycle

        #cfg["QMC"]["statesampling"] = 1
        #for name in cfg["QMC"]:
            #print name, " = ", cfg["QMC"][name]
        cfg["General"]["FFType"] = "plain-full"

        #print "cfg", cfg
        #exit(-1)

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
        muimp = np.real(t_osos)
        U_OOOO = np.real(U_OOOO)

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

        print " "
        print "...................................................."

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
        #print "U_OOOO.shape", U_OOOO.shape
        solver.umatrix = U_OOOO

        ### solve impurity problem 
        mccfgcontainer = []
        iter_no = 1
	### for real
	result = solver.solve(iter_no, mccfgcontainer)
	### for complex
        #result = solver.solve(mccfgcontainer)

        gtau = result.other["gtau-full"]

        ### here comes the function for conversion w2dyn --> triqs
        from converters import w2dyn_ndarray_to_triqs_BlockGF_tau_beta_ntau
        self.G_tau, self.G_tau_error = w2dyn_ndarray_to_triqs_BlockGF_tau_beta_ntau(gtau, self.n_tau, self.beta, self.gf_struct)

        ### I will use the FFT from triqs here...
        for name, g in self.G_tau:
            bl_size = g.target_shape[0]
            known_moments = np.zeros((4, bl_size, bl_size), dtype=np.complex)
            for i in range(bl_size):
                known_moments[1,i,i] = 1

            self.G_iw[name].set_from_fourier(g, known_moments)
