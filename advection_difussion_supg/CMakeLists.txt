CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

PROJECT (advection_diffusion_supg)

FIND_PACKAGE(deal.II 8.0 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

FIND_LIBRARY(boomerAMG_solver_lib libBoomerAMG_solver.so HINTS ../BoomerAMG_solver/lib NO_DEFAULT_PATH)

IF (NOT boomerAMG_solver_lib)
	MESSAGE("*** Could not locate the library libBoomerAMG_solver***")
ENDIF()

FIND_PATH(boomerAMG_solver_include BoomerAMG_solver.h HINTS ../BoomerAMG_solver/source NO_DEFAULT_PATH)

IF (NOT boomerAMG_solver_include)
	MESSAGE("*** Could not locate the libBoomerAMG_solver header file ***")
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()

ADD_SUBDIRECTORY(source)

set_target_properties( advection_diffusion_supg PROPERTIES
RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/bin
RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/bin
)

