# Copy h5 files to binary dir
file(GLOB all_h5_files *.h5)
file(COPY ${all_h5_files} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

set(all_tests exctractor_test SIAM_Discrete_Bath)

foreach(t ${all_tests})
  add_test(NAME ${t} COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/${t}.py)
endforeach()

# Set the PythonPath for all tests
set_property(TEST ${all_tests} PROPERTY ENVIRONMENT
  PATH=${CMAKE_BINARY_DIR}/w2dynamics/cpp:$ENV{PATH}
  PYTHONPATH=${CMAKE_SOURCE_DIR}/python:${CMAKE_BINARY_DIR}/w2dyn_project/src:$ENV{PYTHONPATH}
)
