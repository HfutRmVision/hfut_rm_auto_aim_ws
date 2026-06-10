# Locate the g2o libraries
# A general framework for graph optimization.
#
# This module defines
# G2O_FOUND, if false, do not try to link against g2o
# G2O_LIBRARIES, path to the libg2o
# G2O_INCLUDE_DIR, where to find the g2o header files
# G2O_INCLUDE_DIRS, include directories for targets consuming g2o
#
# Niko Suenderhauf <niko@etit.tu-chemnitz.de>
# Adapted by Felix Endres <endres@informatik.uni-freiburg.de>

IF(UNIX)

  #IF(G2O_INCLUDE_DIR AND G2O_LIBRARIES)
    # in cache already
    #  SET(G2O_FIND_QUIETLY TRUE)
    #ENDIF(G2O_INCLUDE_DIR AND G2O_LIBRARIES)

  MESSAGE(STATUS "Searching for g2o ...")
  FIND_PATH(G2O_INCLUDE_DIR
    NAMES g2o/core/base_vertex.h
    PATHS /usr/local /usr ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES include)

  IF (NOT G2O_INCLUDE_DIR)
    FIND_PATH(G2O_LEGACY_INCLUDE_DIR
      NAMES core math_groups types
      PATHS /usr/local /usr ${CMAKE_PREFIX_PATH}
      PATH_SUFFIXES include/g2o include)

    IF (G2O_LEGACY_INCLUDE_DIR)
      GET_FILENAME_COMPONENT(_G2O_INCLUDE_LEAF "${G2O_LEGACY_INCLUDE_DIR}" NAME)
      IF (_G2O_INCLUDE_LEAF STREQUAL "g2o")
        GET_FILENAME_COMPONENT(G2O_INCLUDE_DIR "${G2O_LEGACY_INCLUDE_DIR}" DIRECTORY)
      ELSE()
        SET(G2O_INCLUDE_DIR "${G2O_LEGACY_INCLUDE_DIR}")
      ENDIF()
    ENDIF()
  ENDIF()

  IF (G2O_INCLUDE_DIR)
    MESSAGE(STATUS "Found g2o headers in: ${G2O_INCLUDE_DIR}")
    SET(G2O_INCLUDE_DIRS ${G2O_INCLUDE_DIR})
    IF (EXISTS "${G2O_INCLUDE_DIR}/g2o/EXTERNAL/ceres/fixed_array.h")
      SET(G2O_DEFINITIONS G2O_USE_VENDORED_CERES)
    ENDIF()
  ENDIF ()

  FIND_LIBRARY(G2O_CORE_LIB             
    NAMES g2o_core g2o_core_rd
    PATHS /usr/local /usr ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES lib lib/${CMAKE_LIBRARY_ARCHITECTURE})
  FIND_LIBRARY(G2O_STUFF_LIB            
    NAMES g2o_stuff g2o_stuff_rd
    PATHS /usr/local /usr ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES lib lib/${CMAKE_LIBRARY_ARCHITECTURE})
  FIND_LIBRARY(G2O_TYPES_SLAM2D_LIB     
    NAMES g2o_types_slam2d g2o_types_slam2d_rd
    PATHS /usr/local /usr ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES lib lib/${CMAKE_LIBRARY_ARCHITECTURE})
  FIND_LIBRARY(G2O_TYPES_SLAM3D_LIB     
    NAMES g2o_types_slam3d g2o_types_slam3d_rd
    PATHS /usr/local /usr ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES lib lib/${CMAKE_LIBRARY_ARCHITECTURE})
  FIND_LIBRARY(G2O_TYPES_SBA_LIB
    NAMES g2o_types_sba g2o_types_sba_rd
    PATHS /usr/local /usr ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES lib lib/${CMAKE_LIBRARY_ARCHITECTURE})
  FIND_LIBRARY(G2O_SOLVER_CHOLMOD_LIB   
    NAMES g2o_solver_cholmod g2o_solver_cholmod_rd
    PATHS /usr/local /usr ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES lib lib/${CMAKE_LIBRARY_ARCHITECTURE})
  FIND_LIBRARY(G2O_SOLVER_PCG_LIB       
    NAMES g2o_solver_pcg g2o_solver_pcg_rd
    PATHS /usr/local /usr ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES lib lib/${CMAKE_LIBRARY_ARCHITECTURE})
  FIND_LIBRARY(G2O_SOLVER_CSPARSE_LIB   
    NAMES g2o_solver_csparse g2o_solver_csparse_rd
    PATHS /usr/local /usr ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES lib lib/${CMAKE_LIBRARY_ARCHITECTURE})
  FIND_LIBRARY(G2O_SOLVER_DENSE_LIB
    NAMES g2o_solver_dense g2o_solver_dense_rd
    PATHS /usr/local /usr ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES lib lib/${CMAKE_LIBRARY_ARCHITECTURE})
  FIND_LIBRARY(G2O_INCREMENTAL_LIB      
    NAMES g2o_incremental g2o_incremental_rd
    PATHS /usr/local /usr ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES lib lib/${CMAKE_LIBRARY_ARCHITECTURE})
  FIND_LIBRARY(G2O_CSPARSE_EXTENSION_LIB
    NAMES g2o_csparse_extension g2o_csparse_extension_rd
    PATHS /usr/local /usr ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES lib lib/${CMAKE_LIBRARY_ARCHITECTURE})

  # (no debug output) build list of libraries present on the filesystem

  # Build a list of only the libraries that actually exist. Some g2o
  # installations build optional components (cholmod, incremental, ...)
  # so only append the entries that are real files.
  SET(G2O_LIBRARIES)
  MACRO(_append_if_exists var)
    IF(EXISTS "${${var}}")
      LIST(APPEND G2O_LIBRARIES "${${var}}")
    ENDIF()
  ENDMACRO()

  _append_if_exists(G2O_CSPARSE_EXTENSION_LIB)
  _append_if_exists(G2O_CORE_LIB)
  _append_if_exists(G2O_STUFF_LIB)
  _append_if_exists(G2O_TYPES_SLAM2D_LIB)
  _append_if_exists(G2O_TYPES_SLAM3D_LIB)
  _append_if_exists(G2O_TYPES_SBA_LIB)
  _append_if_exists(G2O_SOLVER_CHOLMOD_LIB)
  _append_if_exists(G2O_SOLVER_PCG_LIB)
  _append_if_exists(G2O_SOLVER_CSPARSE_LIB)
  _append_if_exists(G2O_SOLVER_DENSE_LIB)
  _append_if_exists(G2O_INCREMENTAL_LIB)

  # Consider G2O found only when headers and at least the two core libs
  # are available; other optional libs are useful but not required.
  IF(G2O_INCLUDE_DIR AND EXISTS "${G2O_CORE_LIB}" AND EXISTS "${G2O_STUFF_LIB}")
    # require core and stuff presence for reliable linking
      SET(G2O_FOUND "YES")
      IF(NOT G2O_FIND_QUIETLY)
        MESSAGE(STATUS "Found libg2o: ${G2O_LIBRARIES}")
      ENDIF()
    ELSE()
    IF(NOT G2O_LIBRARIES)
      IF(G2O_FIND_REQUIRED)
        message(FATAL_ERROR "Could not find libg2o!")
      ENDIF()
    ENDIF()

    IF(NOT G2O_INCLUDE_DIR)
      IF(G2O_FIND_REQUIRED)
        message(FATAL_ERROR "Could not find g2o include directory!")
      ENDIF()
    ENDIF()
  ENDIF()

ENDIF(UNIX)
