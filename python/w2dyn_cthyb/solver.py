from pytriqs.gf import MeshImTime,MeshImFreq, BlockGf
import numpy as np
import os, sys

import auxiliaries.CTQMC

### here come the necessary imports form w2dyn dmft loop
import dmft.impurity as impurity
import auxiliaries.config as config

class Solver():
    
    def __init__(self, beta, gf_struct, n_iw=1025, n_tau=10001, n_l=30):

        self.beta = beta
        self.gf_struct= gf_struct
        self.n_iw = n_iw
        self.n_tau = n_tau
        self.n_l = n_l

        self.tau_mesh = MeshImTime(beta, 'Fermion', n_tau)
        self.Delta_tau = BlockGf(mesh=self.tau_mesh, gf_struct=gf_struct)

        ### 
        #print "len:", len(gf_struct)
        #exit()
        if len(gf_struct) != 2:
            raise Exception("For now gf_struct has to contain exactly 2 blocks!")

    def solve(self, **params_kw):

        depr_params = dict(
            measure_g_tau='measure_G_tau',
            measure_g_l='measure_G_l',
            )

        for key in depr_params.keys():
            if key in params_kw.keys():
                print 'WARNING: cthyb.solve parameter %s is deprecated use %s.' % \
                    (key, depr_params[key])
                val = params_kw.pop(key)
                params_kw[depr_params[key]] = val

        #n_cycles = params_kw.pop("n_cycles", True)
        #print "params_kw", params_kw
        self.n_cycles = params_kw.pop("n_cycles")  ### what does the True or False mean?
        self.measure_G_l = params_kw.pop("measure_G_l")
        self.n_warmup_cycles = params_kw.pop("n_warmup_cycles")
        self.length_cycle = params_kw.pop("length_cycle")
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
        U_OOOO = quartic_tensor_from_operator(self.h_int, fundamental_operators, perm_sym=False)

        ### the U tensor is not correct!
        #print "U_OOOO", U_OOOO
        #for i in range(0,2):
            #for j in range(0,2):
                #for k in range(0,2):
                    #for l in range(0,2):
                        #print i,j,k,l, U_OOOO[i,j,k,l]
        
        #exit()

        ### TODO fix generation of U-Matrix; here it is hardcoded for U=1
        U_OOOO[...] = 0.0
        U_OOOO[0,1,0,1] = 1.0
        U_OOOO[1,0,1,0] = 1.0

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

        

        #exit(-1)


        ### now comes w2dyn!
        import dmft.impurity as impurity
        import auxiliaries.config as config

        ### read w2dyn parameter file; later we will replace this by a 
        ### converter of triqs-parameters to w2dyn-parameters
        cfg_file_name="Parameters.in"
        key_value_args={}
        cfg =  config.get_cfg(cfg_file_name, key_value_args, err=sys.stderr)
        cfg["QMC"]["offdiag"] = 1

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

        #for name in cfg["QMC"]:
            #print name, " = ", cfg["QMC"][name]

        #print "cfg", cfg
        #exit(-1)

        ### initialize the solver; it needs the config-string
        Nseed = 1
        use_mpi = False
        mpi_comm = None
        solver = impurity.CtHybSolver(cfg, Nseed, 0,0,0, not use_mpi, mpi_comm)

        ### generate dummy input that we don't necessarily need
        niw     = 2*cfg["QMC"]["Niw"]
        g0inviw = np.zeros(shape=(self.n_iw, norb, 2, norb, 2))
        fiw     = np.zeros(shape=(self.n_iw, norb, 2, norb, 2))
        fmom    = np.zeros(shape=(2, norb, 2, norb, 2))
        symmetry_moves = ()
        paramag = True
        atom = config.atomlist_from_cfg(cfg, norb)[0]

        ### we begin with real not complex calculations
        g0inviw = np.real(g0inviw)
        fiw = np.real(fiw)
        fmom = np.real(fmom)
        ftau = np.real(ftau)
        muimp = np.real(t_osos)
        U_OOOO = np.real(U_OOOO)

        #print "ftau.shape", ftau.shape
        #print "ftau", ftau
        ### I save it to compare with w2dyn-dmft tools
        np.savetxt("ftau_00.dat",ftau[:,0,0,0,0])
        np.savetxt("ftau_11.dat",ftau[:,0,1,0,1])

        ### here the properties of the impurity will be defined
        imp_problem = impurity.ImpurityProblem(
                        self.beta, g0inviw, fiw, fmom, ftau,
                        muimp, atom.dd_int, None, None, symmetry_moves,
                        paramag)

        print " "
        print "...................................................."
        ### feed impurity problem into solver
        solver.set_problem(imp_problem)

        ### overwrite dummy umatrix in solver class
        #print "solver.umatrix.shape", solver.umatrix.shape
        #print "U_OOOO.shape", U_OOOO.shape
        solver.umatrix = U_OOOO

        ### solve impurity problem 
        mccfgcontainer = []
        iter_no = 1
        result = solver.solve(iter_no, mccfgcontainer)

        gtau = result.other["gtau-full"]
        n_tau = gtau.shape[-1]

        tau_mesh = MeshImTime(self.beta, 'Fermion', n_tau)
        self.G_tau = BlockGf(mesh=tau_mesh, gf_struct=self.gf_struct)

        giw = result.giw
        giw = giw.transpose(1,2,3,4,0)
        n_iw = giw.shape[-1]/2

        iw_mesh = MeshImFreq(self.beta, 'Fermion', n_iw)
        self.G_iw = BlockGf(mesh=iw_mesh, gf_struct=self.gf_struct)
    
        for spin, (name, g_tau) in enumerate(self.G_tau):

            ### in w2dyn the diagonal of the GF is positive
            g_tau.data[:] = -np.transpose(gtau[:, spin, :, spin, :], (2, 0, 1)) 

        for spin, (name, g_iw) in enumerate(self.G_iw):

            ### in w2dyn the diagonal of the GF is positive
            g_iw.data[:] = np.transpose(giw[:, spin, :, spin, :], (2, 0, 1)) 
