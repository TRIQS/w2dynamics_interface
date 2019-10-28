add_subdirectory(cmake)

if(NOT CMAKE_INSTALL_PREFIX STREQUAL TRIQS_ROOT)

  configure_file(w2dynamics_interface.modulefile.in w2dynamics_interface.modulefile @ONLY)
  configure_file(w2dynamics_interfacevars.sh.in w2dynamics_interfacevars.sh @ONLY)

  install(
    FILES
      ${CMAKE_CURRENT_BINARY_DIR}/w2dynamics_interface.modulefile
      ${CMAKE_CURRENT_BINARY_DIR}/w2dynamics_interfacevars.sh
    DESTINATION share
  )
 
  message(STATUS "***************************************************************")
  message(STATUS "* Custom install Location. Use:                                ")
  message(STATUS "*                                                              ")
  message(STATUS "*   source ${CMAKE_INSTALL_PREFIX}/share/w2dynamics_interfacevars.sh      ")
  message(STATUS "*                                                              ")
  message(STATUS "* to set up the environment variables                          ")
  message(STATUS "***************************************************************")

endif()
