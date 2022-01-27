# Copy h5 files to binary dir
file(GLOB_RECURSE all_h5_ref_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.ref.h5)
foreach(file ${all_h5_ref_files})
  configure_file(${file} ${file} COPYONLY)
endforeach()

# List of all tests
set(all_tests exctractor_test SIAM_Discrete_Bath 2orb_Discrete_Bath 2orb_Discrete_Bath_2pGF)

try_compile(USE_LIBCXX ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/check_libcxx.cpp)
set(GCCVER_GE11 OFF)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 11)
  set(GCCVER_GE11 ON)
endif()
foreach(test ${all_tests})
  get_filename_component(test_name ${test} NAME_WE)
  get_filename_component(test_dir ${test} DIRECTORY)
  add_test(NAME Py_${test_name}
    COMMAND ${TRIQS_PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${test_dir}/${test_name}.py
      $<$<BOOL:${USE_LIBCXX}>:--libcxx>
      $<$<BOOL:${GCCVER_GE11}>:--gccver_ge11>
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_dir}
  )
  set_property(TEST Py_${test_name} APPEND PROPERTY ENVIRONMENT PYTHONPATH=${PROJECT_BINARY_DIR}/python:${W2DYN_ROOT}:$ENV{PYTHONPATH})
endforeach()
