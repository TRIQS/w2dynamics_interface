from triqs.gf import *
from h5 import HDFArchive
from triqs.plot.mpl_interface import oplot, plt

estimators = ['dyson', 'improved_worm', 'symmetric_improved_worm']

fig, ax = plt.subplots(2, 2, figsize=(12, 7))
fig.suptitle("Single orbital AIM with two bath sites in a magnetic field")

with HDFArchive('improved_estimators_solution.h5', 'r') as ar:
    for selfenergy in estimators:
        plt.sca(ax[0,0])
        oplot(ar[selfenergy]['G_iw']['up'], '.-', mode='R', label=selfenergy)
        plt.sca(ax[0,1])
        oplot(ar[selfenergy]['G_iw']['up'], '.-', mode='I', label=selfenergy)
        plt.sca(ax[1,0])
        oplot(ar[selfenergy]['Sigma_iw']['up'], '.-', mode='R', label=selfenergy)
        plt.sca(ax[1,1])
        oplot(ar[selfenergy]['Sigma_iw']['up'], '.-', mode='I', label=selfenergy)

    # The Hartree shift is the zeroth moment of the self-energy, which the
    # solver takes from the density matrix instead of from Sigma itself
    Sigma_Hartree = ar['dyson']['Sigma_moments']['up'][0][0, 0]
    ax[1,0].axhline(Sigma_Hartree, color='k', ls='--',
                    label=r'$\Sigma_\mathrm{Hartree}$')

ax[0,0].set_ylabel(r'$\operatorname{Re} G_\uparrow(i\omega_n)$')
ax[0,0].legend(loc='upper right')
ax[0,1].set_ylabel(r'$\operatorname{Im} G_\uparrow(i\omega_n)$')
ax[0,1].legend(loc='upper right')
ax[1,0].set_ylabel(r'$\operatorname{Re} \Sigma_\uparrow(i\omega_n)$')
ax[1,0].legend(loc='lower left')
ax[1,1].set_ylabel(r'$\operatorname{Im} \Sigma_\uparrow(i\omega_n)$')
ax[1,1].legend(loc='lower left')
plt.tight_layout()
plt.savefig("improved_estimators_plot.pdf", bbox_inches="tight")
