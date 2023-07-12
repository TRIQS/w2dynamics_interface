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

cmake_minimum_required(VERSION 3.12.4 FATAL_ERROR)
cmake_policy(VERSION 3.12.4)
if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

# ############
# Define Project
project(w2dynamics_interface VERSION 3.1.0 LANGUAGES C CXX)
get_directory_property(IS_SUBPROJECT PARENT_DIRECTORY)

# ############
# Load TRIQS and CPP2PY
find_package(TRIQS 3.1 REQUIRED)

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
if(NOT IS_SUBPROJECT)
  message(STATUS "-------- CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX} --------")
endif()
set(${PROJECT_NAME}_BINARY_DIR ${PROJECT_BINARY_DIR} CACHE STRING "Binary directory of the ${PROJECT_NAME} Project")

# Make additional Find Modules available
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/share/cmake/Modules)

# ############
# CMake Options

# Default to Release build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Type of build" FORCE)
endif()
message(STATUS "-------- BUILD-TYPE: ${CMAKE_BUILD_TYPE} --------")


# --- Define the w2dynamics interface version numbers and get the git hash
set(W2DYN_GIT_HASH b5a2bcc9ecdb694703a4e683b63352bedf116882)
message(STATUS "w2dynamics git hash: ${W2DYN_GIT_HASH}")

# --- Build w2dynamics as an external project
get_property(FFTW_LIBRARIES TARGET triqs::fftw PROPERTY INTERFACE_LINK_LIBRARIES)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
# convert error in warning due to f2py incompatibility
set(W2DYN_CFLAGS "${CMAKE_C_FLAGS} -Wno-error=incompatible-function-pointer-types")
message("Detected Clang version 16 or higher. Suppress incompatible function pointer type warnings for clang.")
endif()

include(ExternalProject)
ExternalProject_Add(w2dynamics
  PREFIX w2dyn_project
  #GIT_REPOSITORY https://github.com/w2dynamics/w2dynamics
  GIT_REPOSITORY https://github.com/the-hampel/w2dynamics
  GIT_TAG ${W2DYN_GIT_HASH}
  GIT_CONFIG advice.detachedHead=false
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
             -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
             -DPYTHON_EXECUTABLE=${TRIQS_PYTHON_EXECUTABLE}
             -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
             -DFFTW_LIBRARIES=${FFTW_LIBRARIES}
             -DCMAKE_C_FLAGS=${W2DYN_CFLAGS}
  INSTALL_COMMAND ""
  PATCH_COMMAND ${TRIQS_GIT_EXECUTABLE} checkout . && ${TRIQS_GIT_EXECUTABLE} apply ${PROJECT_SOURCE_DIR}/w2dynamics.patch
)

set(W2DYN_ROOT ${PROJECT_BINARY_DIR}/w2dyn_project/src/w2dynamics)
install(DIRECTORY ${W2DYN_ROOT}/w2dyn DESTINATION ${TRIQS_PYTHON_LIB_DEST_ROOT} FILES_MATCHING PATTERN "*.py" PATTERN "*.so" PATTERN "configspec")
if(NOT IS_SUBPROJECT)
  message(STATUS "-------- BUILD-TYPE: ${CMAKE_BUILD_TYPE} --------")
endif()

# Python Support
if(NOT TRIQS_WITH_PYTHON_SUPPORT)
  message(FATAL_ERROR "TRIQS was installed without Python support. Cannot build the Python Interface. Disable the build with -DPythonSupport=OFF")
endif()

# Documentation
option(Build_Documentation "Build documentation" OFF)

# Testing
option(Build_Tests "Build tests" ON)
if(Build_Tests)
  enable_testing()
endif()

# Build static libraries by default
option(BUILD_SHARED_LIBS "Enable compilation of shared libraries" OFF)

# ############
# Global Compilation Settings

# Export the list of compile-commands into compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Disable compiler extensions
set(CMAKE_CXX_EXTENSIONS OFF)

# Provide additional debugging information for Debug builds
add_compile_options($<$<CONFIG:Debug>:-ggdb3>)

# Create an Interface target for compiler warnings
add_library(${PROJECT_NAME}_warnings INTERFACE)
target_compile_options(${PROJECT_NAME}_warnings
  INTERFACE
    -Wall
    -Wextra
    -Wpedantic
    -Wno-sign-compare
    $<$<CXX_COMPILER_ID:GNU>:-Wno-comma-subscript>
    $<$<CXX_COMPILER_ID:GNU>:-Wshadow=local>
    $<$<CXX_COMPILER_ID:GNU>:-Wno-attributes>
    $<$<CXX_COMPILER_ID:Clang>:-Wno-deprecated-comma-subscript>
    $<$<CXX_COMPILER_ID:Clang>:-Wno-unknown-warning-option>
    $<$<CXX_COMPILER_ID:Clang>:-Wshadow>
    $<$<CXX_COMPILER_ID:Clang>:-Wno-gcc-compat>
    $<$<CXX_COMPILER_ID:Clang>:-Wno-c++20-extensions>
    $<$<CXX_COMPILER_ID:AppleClang>:-Wno-deprecated-comma-subscript>
    $<$<CXX_COMPILER_ID:AppleClang>:-Wno-unknown-warning-option>
    $<$<CXX_COMPILER_ID:AppleClang>:-Wshadow>
    $<$<CXX_COMPILER_ID:AppleClang>:-Wno-gcc-compat>
    $<$<CXX_COMPILER_ID:AppleClang>:-Wno-c++20-extensions>
)

# #############
# Build Project

# Find / Build dependencies
add_subdirectory(deps)

# Tests
if(Build_Tests)
  add_subdirectory(test)
endif()

# Python
add_subdirectory(python/w2dyn_cthyb)

# Docs
if(Build_Documentation)
  add_subdirectory(doc)
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
  set(CPACK_PACKAGE_NAME ${PROJECT_NAME})
  set(CPACK_GENERATOR "DEB")
  set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
  set(CPACK_PACKAGE_CONTACT "https://github.com/TRIQS/${PROJECT_NAME}")
  execute_process(COMMAND dpkg --print-architecture OUTPUT_VARIABLE CMAKE_DEBIAN_PACKAGE_ARCHITECTURE OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(CPACK_DEBIAN_PACKAGE_DEPENDS "triqs (>= 3.1)")
  set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
  set(CPACK_DEBIAN_PACKAGE_GENERATE_SHLIBS ON)
  include(CPack)
endif()
