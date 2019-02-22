#source/CMakeLists.txt

SET(CMAKE_INCLUDE_CURRENT_DIR ON)

SET(SOURCE_LIST BoomerAMG_solver.cc)
#LIST(APPEND SOURCE_LIST next_file_if_needed.cpp)

ADD_LIBRARY(BoomerAMG_solver SHARED ${SOURCE_LIST})
DEAL_II_SETUP_TARGET(BoomerAMG_solver)
