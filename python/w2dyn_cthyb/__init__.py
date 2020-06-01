from triqs.utility import mpi

from solver import Solver
from converters import get_test_impurity_model,NO_to_Nos,triqs_gf_to_w2dyn_ndarray_g_tosos_beta_ntau,triqs_gf_to_w2dyn_ndarray_g_wosos_beta_niw,generate_testimpurity_with_triqs


__all__ = ['Solver','get_test_impurity_model','NO_to_Nos','triqs_gf_to_w2dyn_ndarray_g_tosos_beta_ntau','triqs_gf_to_w2dyn_ndarray_g_wosos_beta_niw','generate_testimpurity_with_triqs']

BANNER = r"""
           ______
         _/XXXXXX\ ___ __  ____   /|            W2DYNAMICS - Wuerzburg/Wien strong
  |\    | |X/  \X| __ \\ \/ /  \ | |                 coupling impurity solver
  \ \ |\| |____/X| | \ \\  /| \ \| |
   \ \\ \ /XXXXXX| |_/ // / | |\ \ |   AUTHORS: M Wallerberger, A Hausoel, P Gunacker, A Kowalski, 
    \__\__|X/____|____//_/  |_| \__|            N Parragh, F Goth, K Held and G Sangiovanni
         |XXXXXXXX|                    VERSION: %s (%s)


                    &----------------------------------------------------&
                    | IF W2DYN WAS OF USE FOR A PUBLICATION, PLEASE CITE |
                    |      https://doi.org/10.1016/j.cpc.2018.09.007     |
                    |          https://arxiv.org/abs/1807.00361          |
                    &----------------------------------------------------&

"""

CODE_VERSION = 1, 0, "0"
CODE_VERSION_STRING = ".".join(map(str,CODE_VERSION))
CODE_DATE = "July 2018"
OUTPUT_VERSION = 2, 2

if mpi.is_master_node(): print BANNER % (CODE_VERSION_STRING, CODE_DATE)
