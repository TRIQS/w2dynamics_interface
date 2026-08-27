.. _improved_estimators:

Improved estimators for the self-energy
=======================================

By default the solver measures the Green's function and the self-energy follows
from the Dyson equation,

.. math::

  \Sigma(i \omega_n) = G_0^{-1}(i \omega_n) - G^{-1}(i \omega_n).

Because the Green's function enters inverted, its statistical and systematic
errors are amplified with the square of the frequency, which makes the tail of
the self-energy the least reliable part of it. The improved estimators avoid
this by worm sampling a quantity that already contains the interaction. The
``selfenergy`` parameter of ``solve()`` chooses between them:

``dyson``
  Measure the Green's function and use the Dyson equation. This is the default.

``improved_worm``
  Worm sample :math:`G \Sigma` and construct the Green's function from it,
  Eq. (9) of `PRB 100, 075119 (2019) <https://doi.org/10.1103/PhysRevB.100.075119>`_.

``symmetric_improved_worm``
  Worm sample the symmetric combination and construct the Green's function from
  it, Eq. (13) of the same paper.

Both improved estimators build the Green's function on top of the bare
propagator, so it carries the exact asymptotics of :math:`G_0` and the tail of
the self-energy no longer suffers from the truncated representation of a
measured Green's function. The self-energy itself is still taken from the Dyson
equation in all three cases.

Let us compare the three on one impurity orbital in a magnetic field coupled to
two bath sites. Here is the python :download:`script <improved_estimators.py>`:

.. literalinclude:: improved_estimators.py

Running this script on a single processor takes about one and a half minutes and
generates an HDF5 archive file called :file:`improved_estimators_solution.h5`
holding the Green's function and the self-energy of each estimator. Let us plot
them:

.. plot:: guide/improved_estimators_plot.py
   :include-source:
   :scale: 70

The three Green's functions are indistinguishable, as they have to be. The
self-energies agree at low frequency, where they are needed most, but differ in
the tail, which has to approach the constant Hartree shift. Averaged over the
last twenty Matsubara frequencies of the spin-up component:

============================= =================
estimator                     :math:`\mathrm{Re}\, \Sigma` in the tail
============================= =================
``dyson``                     3.01 :math:`\pm` 0.04
``improved_worm``             2.65 :math:`\pm` 0.19
``symmetric_improved_worm``   2.76 :math:`\pm` 0.01
============================= =================

The Dyson result is smooth but drifts upwards by 0.25 over the plotted window,
six times its own scatter: a systematic error, inherited from the truncated
Legendre representation of the Green's function and amplified by the inversion.
Both improved estimators are flat instead, and the symmetric one is flat to
within 0.01. The plain improved estimator is unbiased as well, but its scatter
is an order of magnitude larger, so the symmetric one is the better choice
unless there is a reason to prefer :math:`G \Sigma` itself.

Restrictions
------------

The improved estimators imply worm sampling, which has consequences beyond the
self-energy:

* The Green's function in imaginary time and in the Legendre basis are not
  available, because they are accumulated in the partition function space that
  worm sampling does not normalize. Only ``G_iw`` and ``Sigma_iw`` are set.

* The hybridization function has to be diagonal. w2dynamics does not implement
  worm sampling above the one-particle sector for an offdiagonal hybridization,
  and the interface sets ``offdiag = 1`` as soon as a block of ``gf_struct`` is
  larger than 1x1, so a multi-orbital impurity has to be set up with one block
  per spin-orbital.

* ``improved_worm`` is refused for more than one orbital. w2dynamics normalizes
  it with the sparsity of a single-orbital interaction matrix, which makes the
  result wrong by a constant factor for any other interaction. Use
  ``symmetric_improved_worm``, which does not have this defect.
