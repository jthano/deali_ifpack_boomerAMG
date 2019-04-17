#src/CMakeLists.txt
#
#SET(CMAKE_INCLUDE_CURRENT_DIR ON)

ADD_EXECUTABLE(advection_diffusion_supg supg.cc)

TARGET_LINK_LIBRARIES(advection_diffusion_supg ${boomerAMG_solver_lib})
TARGET_INCLUDE_DIRECTORIES(advection_diffusion_supg PRIVATE ${boomerAMG_solver_include})

DEAL_II_SETUP_TARGET(advection_diffusion_supg)

