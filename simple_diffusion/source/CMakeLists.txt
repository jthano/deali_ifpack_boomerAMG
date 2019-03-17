#src/CMakeLists.txt
#
#SET(CMAKE_INCLUDE_CURRENT_DIR ON)

ADD_EXECUTABLE(simple_diffusion amg_elliptic.cc)

TARGET_LINK_LIBRARIES(simple_diffusion ${boomerAMG_solver_lib})

DEAL_II_SETUP_TARGET(simple_diffusion)

