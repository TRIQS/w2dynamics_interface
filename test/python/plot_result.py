from pytriqs.gf import *
from pytriqs.archive import HDFArchive

with HDFArchive("aim_solution_w2dyn.h5",'r') as Results:
    G_tau = Results["G_tau"]
    G_iw = Results["G_iw"]

from pytriqs.plot.mpl_interface import oplot, oploti, oplotr, plt
plt.figure()
oplot(G_tau)
plt.figure()
oplot(G_iw)

plt.show()
