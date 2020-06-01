from triqs.gf import *
from h5 import *
from triqs.plot.mpl_interface import oplot, oploti, oplotr, plt
import numpy as np

ar_w2dyn = HDFArchive('aim_solution_w2dyn.h5','r')
ar_triqs = HDFArchive('aim_solution_triqs.h5','r')

beta = ar_triqs["G_iw/down/mesh/domain/beta"]
print "beta", beta

### plot Matsubara GF
oplot(ar_triqs['G_iw']['up'][0,0], '-', x_window = (-25,25), mode = 'R', name = "Re G$_{triqs}$")
oplot(ar_triqs['G_iw']['up'][0,0], '-', x_window = (-25,25), mode = 'I', name = "Re G$_{triqs}$")
oplot(ar_w2dyn['G_iw']['up'][0,0], '-', x_window = (-25,25), mode = 'R', name = "Re G$_{w2dyn}$")
oplot(ar_w2dyn['G_iw']['up'][0,0], '-', x_window = (-25,25), mode = 'I', name = "Re G$_{w2dyn}$")
plt.legend(loc = 'best')
plt.ylabel("G")
plt.show()

