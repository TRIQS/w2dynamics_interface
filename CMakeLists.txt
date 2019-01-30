# Start configuration
cmake_minimum_required(VERSION 2.8.12)
project(w2dynamics_interface Fortran CXX)
set(W2DYNAMICS_INTERFACE_VERSION 1.0.0)
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# Default to Release build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Type of build" FORCE)
endif()
message( STATUS "-------- BUILD-TYPE: ${CMAKE_BUILD_TYPE} --------")

# Load Dependencies
find_package(TRIQS 2.2 REQUIRED)

# Default Install directory to TRIQS_ROOT if not given or invalid.
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT OR (NOT IS_ABSOLUTE ${CMAKE_INSTALL_PREFIX}))
  message(STATUS "No install prefix given (or invalid). Defaulting to TRIQS_ROOT")
  set(CMAKE_INSTALL_PREFIX ${TRIQS_ROOT} CACHE PATH "default install path" FORCE)
endif()
message(STATUS "-------- INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX} --------")

# --- Build w2dynamics as an external project
get_property(BLAS_LAPACK_LIBRARIES TARGET triqs::blas_lapack PROPERTY INTERFACE_LINK_LIBRARIES)
get_property(FFTW_LIBRARIES TARGET triqs::fftw PROPERTY INTERFACE_LINK_LIBRARIES)

include(ExternalProject)
set(W2DYN_ROOT ${CMAKE_INSTALL_PREFIX})
ExternalProject_Add(w2dyn
  PREFIX w2dyn_project
  GIT_REPOSITORY https://github.com/wentzell/w2dynamics
  GIT_TAG cmake_adjustments
  GIT_CONFIG advice.detachedHead=false
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
  	     -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
	     -DPYTHON_EXECUTABLE=${TRIQS_PYTHON_INTERPRETER}
	     -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
	     -DCMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER}
	     -DBLAS_LIBRARIES=${BLAS_LAPACK_LIBRARIES}
	     -DLAPACK_LIBRARIES=${BLAS_LAPACK_LIBRARIES}
	     -DFFTW_LIBRARIES=${FFTW_LIBRARIES}
	     -DNFFT_ROOT=${NFFT_ROOT}
  INSTALL_COMMAND ""
)
set(W2DYN_ROOT ${PROJECT_BINARY_DIR}/w2dyn_project/src/w2dyn)
install(DIRECTORY ${W2DYN_ROOT}/auxiliaries ${W2DYN_ROOT}/dmft ${W2DYN_ROOT}/maxent DESTINATION ${TRIQS_PYTHON_LIB_DEST_ROOT}/w2dyn FILES_MATCHING PATTERN "*.py" PATTERN "*.so" PATTERN "configspec")
install(FILES __init__.py DESTINATION ${TRIQS_PYTHON_LIB_DEST_ROOT}/w2dyn)

# The python interface
add_subdirectory(python/w2dyn_cthyb)

# Tests
option(Build_Tests "Build tests" ON)
if(Build_Tests)
  enable_testing()
  add_subdirectory(test/python)
endif()

# Documentation
option(Build_Documentation "Build documentation" OFF)
if(Build_Documentation)
  find_package(MathJax REQUIRED)
  message(STATUS "Detected MathJax at ${MATHJAX_PATH}")
  if(NOT TRIQS_WITH_DOCUMENTATION)
    message(WARNING " TRIQS library has not been compiled with its documentation!\n Cannot build documentation!")
  else()
    message(STATUS "Documentation Build enabled")
    add_subdirectory(doc)
  endif()
endif()

# Build a debian package
option(BUILD_DEBIAN_PACKAGE "Build a deb package" OFF)
if(BUILD_DEBIAN_PACKAGE)
  if(NOT CMAKE_INSTALL_PREFIX STREQUAL "/usr")
    message(FATAL_ERROR "CMAKE_INSTALL_PREFIX must be /usr for packaging")
  endif()
  SET(CPACK_GENERATOR "DEB")
  SET(CPACK_PACKAGE_VERSION ${W2DYNAMICS_INTERFACE_VERSION})
  SET(CPACK_PACKAGE_CONTACT "https://github.com/TRIQS/w2dynamics_interface")
  EXECUTE_PROCESS(COMMAND dpkg --print-architecture OUTPUT_VARIABLE CMAKE_DEBIAN_PACKAGE_ARCHITECTURE OUTPUT_STRIP_TRAILING_WHITESPACE)
  SET(CPACK_DEBIAN_PACKAGE_DEPENDS "triqs (>= 2.0)")
  SET(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
  SET(CPACK_DEBIAN_PACKAGE_GENERATE_SHLIBS ON)
  INCLUDE(CPack)
endif()
