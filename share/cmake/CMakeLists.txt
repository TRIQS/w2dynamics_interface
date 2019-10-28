configure_file(w2dynamics_interface-config.cmake.in w2dynamics_interface-config.cmake @ONLY)
configure_file(w2dynamics_interface-config-version.cmake.in w2dynamics_interface-config-version.cmake @ONLY)
install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/w2dynamics_interface-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/w2dynamics_interface-config-version.cmake
  DESTINATION lib/cmake/w2dynamics_interface
)
