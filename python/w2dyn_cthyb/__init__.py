################################################################################
#
# w2dynamics_interface - An Interface to the w2dynamics cthyb code
#
# Copyright (C) 2019 by The Simons Foundation
# Authors: Andreas Hausoel, Hugo U. R. Strand, Nils Wentzell
#
# w2dynamics_interface is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# w2dynamics_interface is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# w2dynamics_interface. If not, see <http://www.gnu.org/licenses/>.
#
################################################################################

from triqs.utility import mpi

from .solver import Solver
from .converters import get_test_impurity_model,NO_to_Nos,triqs_gf_to_w2dyn_ndarray_g_tosos_beta_ntau,triqs_gf_to_w2dyn_ndarray_g_wosos_beta_niw,generate_testimpurity_with_triqs

from w2dyn.auxiliaries import BANNER, CODE_VERSION, CODE_VERSION_STRING, CODE_DATE, OUTPUT_VERSION


__all__ = ['Solver','get_test_impurity_model','NO_to_Nos','triqs_gf_to_w2dyn_ndarray_g_tosos_beta_ntau','triqs_gf_to_w2dyn_ndarray_g_wosos_beta_niw','generate_testimpurity_with_triqs']


INTERFACE_BANNER_ADD = r"""

             &----------------------------------------------------&
             | IF W2DYN WAS OF USE FOR A PUBLICATION, PLEASE CITE |
             |      https://doi.org/10.1016/j.cpc.2018.09.007     |
             |          https://arxiv.org/abs/1807.00361          |
             &----------------------------------------------------&

"""

if mpi.is_master_node(): print((BANNER + INTERFACE_BANNER_ADD) % (CODE_VERSION_STRING, CODE_DATE))
