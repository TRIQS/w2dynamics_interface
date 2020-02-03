# ##############################################################################
#
# w2dynamics_interface - An Interface to the w2dynamcis cthyb solver
#
# Copyright (C) 2019 Simons Foundation
#   Authors: N. Wentzell
#
# w2dynamics_interface is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# w2dynamics_interface is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# w2dynamics_interface (in the file COPYING.txt in this directory). If not, see
# <http://www.gnu.org/licenses/>.
#
# ##############################################################################

cmake_minimum_required(VERSION 3.0.2 FATAL_ERROR)
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

# ############
# Define Project
project(W2DYNAMICS_INTERFACE VERSION 2.2.0 LANGUAGES Fortran C CXX)
get_directory_property(IS_SUBPROJECT PARENT_DIRECTORY)

# ############
# Load TRIQS and CPP2PY
find_package(TRIQS 2.2 REQUIRED)
find_package(Cpp2Py 1.6 REQUIRED)

# Get the git hash & print status
triqs_get_git_hash_of_source_dir(PROJECT_GIT_HASH)
message(STATUS "${PROJECT_NAME} version : ${PROJECT_VERSION}")
message(STATUS "${PROJECT_NAME} Git hash: ${PROJECT_GIT_HASH}")

# Enforce Consistent Versioning
if(NOT ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR} VERSION_EQUAL ${TRIQS_VERSION_MAJOR}.${TRIQS_VERSION_MINOR})
  message(FATAL_ERROR "The ${PROJECT_NAME} version ${PROJECT_VERSION} is not compatible with TRIQS version ${TRIQS_VERSION}.")
endif()

# Default Install directory to TRIQS_ROOT if not given or invalid.
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT OR (NOT IS_ABSOLUTE ${CMAKE_INSTALL_PREFIX}))
  message(STATUS "No install prefix given (or invalid). Defaulting to TRIQS_ROOT")
  set(CMAKE_INSTALL_PREFIX ${TRIQS_ROOT} CACHE PATH "default install path" FORCE)
  set(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT FALSE)
endif()
message(STATUS "-------- CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX} --------")

# ############
# Options

# Default to Release build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Type of build" FORCE)
endif()
message(STATUS "-------- BUILD-TYPE: ${CMAKE_BUILD_TYPE} --------")


# --- Define the w2dynamics interface version numbers and get the git hash
set(W2DYN_GIT_HASH 4d530d46fc6dd4583cf0f698ae85a477e73682f0)
message(STATUS "w2dynamics git hash: ${W2DYN_GIT_HASH}")

# --- Build w2dynamics as an external project
get_property(BLAS_LAPACK_LIBRARIES TARGET triqs::blas_lapack PROPERTY INTERFACE_LINK_LIBRARIES)
get_property(FFTW_LIBRARIES TARGET triqs::fftw PROPERTY INTERFACE_LINK_LIBRARIES)

include(ExternalProject)
set(W2DYN_ROOT ${CMAKE_INSTALL_PREFIX})
ExternalProject_Add(w2dyn
  PREFIX w2dyn_project
  GIT_REPOSITORY https://github.com/wentzell/w2dynamics
  GIT_TAG ${W2DYN_GIT_HASH}
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
install(FILES ${W2DYN_ROOT}/__init__.py DESTINATION ${TRIQS_PYTHON_LIB_DEST_ROOT}/w2dyn)

# Tests
option(Build_Tests "Build tests" ON)
if(Build_Tests AND NOT IS_SUBPROJECT)
  enable_testing()
  add_subdirectory(test)
endif()

if(TRIQS_WITH_PYTHON_SUPPORT)
  # Python interface
  add_subdirectory(python/w2dyn_cthyb)

  # Build the documentation
  option(Build_Documentation "Build documentation" OFF)
  if(Build_Documentation AND NOT IS_SUBPROJECT)
    if(NOT TRIQS_WITH_DOCUMENTATION)
      message(WARNING "TRIQS was installed without documentation! Cannot build documentation.")
    else()
      message(STATUS "Documentation Build enabled")
      add_subdirectory(doc)
    endif()
  endif()
else()
  message(WARNING "TRIQS was installed without Python support. Cannot build the Python Interface and Documentation.")
endif()

# Additional configuration files
add_subdirectory(share)

# #############
# Debian Package

option(BUILD_DEBIAN_PACKAGE "Build a deb package" OFF)
if(BUILD_DEBIAN_PACKAGE AND NOT IS_SUBPROJECT)
  if(NOT CMAKE_INSTALL_PREFIX STREQUAL "/usr")
    message(FATAL_ERROR "CMAKE_INSTALL_PREFIX must be /usr for packaging")
  endif()
  set(CPACK_PACKAGE_NAME w2dynamics_interface)
  set(CPACK_GENERATOR "DEB")
  set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
  set(CPACK_PACKAGE_CONTACT "https://github.com/TRIQS/w2dynamics_interface")
  execute_process(COMMAND dpkg --print-architecture OUTPUT_VARIABLE CMAKE_DEBIAN_PACKAGE_ARCHITECTURE OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(CPACK_DEBIAN_PACKAGE_DEPENDS "triqs (>= 2.2)")
  set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
  set(CPACK_DEBIAN_PACKAGE_GENERATE_SHLIBS ON)
  include(CPack)
endif()

