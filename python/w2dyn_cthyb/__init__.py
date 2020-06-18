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
