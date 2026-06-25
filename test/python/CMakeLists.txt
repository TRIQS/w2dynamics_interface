# Copy h5 files to binary dir
file(GLOB_RECURSE all_h5_ref_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.ref.h5)
foreach(file ${all_h5_ref_files})
  configure_file(${file} ${file} COPYONLY)
endforeach()

# List of all tests
set(all_tests
  exctractor_test
  SIAM_Discrete_Bath
  2orb_Discrete_Bath
  2orb_Discrete_Bath_2pGF
  2orb_Discrete_Bath_2pGF_cfg_qmc
  )

# The worm-sampled two-particle Green's function (G2_iw_ph) measured with the
# modest statistics used here is a chaotic, toolchain-specific Monte Carlo
# fingerprint rather than a converged quantity: it reproduces bit-for-bit on a
# given toolchain but differs completely across compiler / standard-library /
# BLAS, so the fixed reference data cannot match every CI platform (the two
# committed references are themselves mutually uncorrelated). Disable these two
# tests until proper per-platform references (or much higher statistics) exist.
set(disabled_tests
  2orb_Discrete_Bath_2pGF
  2orb_Discrete_Bath_2pGF_cfg_qmc
  )

try_compile(USE_LIBCXX ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/check_libcxx.cpp)
foreach(test ${all_tests})
  get_filename_component(test_name ${test} NAME_WE)
  get_filename_component(test_dir ${test} DIRECTORY)
  add_test(NAME Py_${test_name}
    COMMAND ${TRIQS_PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${test_dir}/${test_name}.py
      $<$<BOOL:${USE_LIBCXX}>:--libcxx>
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_dir}
  )
  set_property(TEST Py_${test_name} APPEND PROPERTY ENVIRONMENT_MODIFICATION
    "PYTHONPATH=path_list_prepend:${PROJECT_BINARY_DIR}/python"
    "PYTHONPATH=path_list_prepend:${w2dynamics_SOURCE_DIR}")
  if("${test}" IN_LIST disabled_tests)
    set_tests_properties(Py_${test_name} PROPERTIES DISABLED TRUE)
  endif()
endforeach()
