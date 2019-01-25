from pytriqs.gf import *
from pytriqs.operators import *
from pytriqs.archive import HDFArchive
import pytriqs.utility.mpi as mpi
import numpy as np

#w2dyn=False
w2dyn=True

### here comes the solver
if w2dyn:
   from w2dyn_cthyb import Solver
else:
   from triqs_cthyb import Solver

### Parameters
U = 1.0
beta = 100.0
#e_f = 0.5
e_f = -U/2.0

### a discrete AIM
e1, e2 = -0.43, 0.134
V1, V2 = 0.38, 0.38

### Construct the impurity solver with the inverse temperature
### and the structure of the Green's functions
S = Solver(beta = beta, gf_struct = [ ['up',[0]], ['down',[0]] ], n_l = 30, n_iw=2000,  n_tau=4002)

### the hybridistation function in Matsubara
delta_iw_block = ( V1**2 * inverse( iOmega_n - e1 ) + V2**2 * inverse( iOmega_n - e2 ) )

if w2dyn:
    ### generate a Delta(tau) for w2dyn
    iw_mesh = MeshImFreq(beta, 'Fermion', S.n_iw)
    Delta_iw = BlockGf(mesh=iw_mesh, gf_struct=S.gf_struct)

    for name, d in Delta_iw: 
        d << delta_iw_block

        #d << Fourier(Delta_iw[name])   # this is the same as below, but blockwise

    ### we need the object for holes
    from pytriqs.gf.tools import conjugate
    Delta_iw = conjugate(Delta_iw)

    ### Fourier Transform
    S.Delta_tau << Fourier(Delta_iw)

    ### Write out F(iw) for comparison with w2dyn_dmft tools
    from example import triqs_gf_to_w2dyn_ndarray_g_wosos_beta_niw
    fiw, _, __ = triqs_gf_to_w2dyn_ndarray_g_wosos_beta_niw(Delta_iw)
    data = np.column_stack((np.real(fiw[:,0,0,0,0]),np.imag(fiw[:,0,0,0,0])))
    np.savetxt("fiw_00.dat",data)
    data = np.column_stack((np.real(fiw[:,0,1,0,1]),np.imag(fiw[:,0,1,0,1])))
    np.savetxt("fiw_11.dat",data)
    #exit()

else:

   # Initialize the non-interacting Green's function S.G0_iw
   for name, g0 in S.G0_iw: 
       g0 << inverse ( iOmega_n - e_f - delta_iw_block )
       
   #from pytriqs.plot.mpl_interface import oplot, oploti, oplotr, plt
   #oplot(S.G0_iw)
   #plt.show()

   ### Write out F(iw) for comparison with w2dyn_dmft tools
   from example import triqs_gf_to_w2dyn_ndarray_g_wosos_beta_niw
   G0_iw, _, __ = triqs_gf_to_w2dyn_ndarray_g_wosos_beta_niw(S.G0_iw)
   data = np.column_stack((np.real(G0_iw[:,0,0,0,0]),np.imag(G0_iw[:,0,0,0,0])))
   np.savetxt("G0_triqs_00.dat",data)
   data = np.column_stack((np.real(G0_iw[:,0,1,0,1]),np.imag(G0_iw[:,0,1,0,1])))
   np.savetxt("G0_triqs_11.dat",data)

   #exit()
    

### Run the solver. The results will be in S.G_tau, S.G_iw and S.G_l
if w2dyn:
    S.solve(h_int = U * n('up',0) * n('down',0) - e_f* ( n('up',0) + n('down',0) ),     # Local Hamiltonian + quadratic terms
            n_cycles  = 5000,                      # Number of QMC cycles
            length_cycle = 100,                      # Length of one cycle
            n_warmup_cycles = 5000,                 # Warmup cycles
            measure_G_l = False)                      # Measure G_l
else:
    S.solve(h_int = U * n('up',0) * n('down',0) ,     # Local Hamiltonian
            n_cycles  = 5000,                      # Number of QMC cycles
            length_cycle = 100,                      # Length of one cycle
            n_warmup_cycles = 5000,                 # Warmup cycles
            measure_G_l = False)                      # Measure G_l

### plot Greens function
#from pytriqs.plot.mpl_interface import oplot, oploti, oplotr, plt
#oplot(S.G_tau)
#plt.show()

### write stuff to file
if w2dyn:
   with HDFArchive("aim_solution_w2dyn.h5",'w') as Results:
       Results["G_tau"] = S.G_tau
       Results["G_iw"] = S.G_iw
else:
   with HDFArchive("aim_solution_triqs.h5",'w') as Results:
       Results["G_tau"] = S.G_tau
       Results["G_iw"] = S.G_iw
