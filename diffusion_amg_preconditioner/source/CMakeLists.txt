#src/CMakeLists.txt
#
#SET(CMAKE_INCLUDE_CURRENT_DIR ON)

ADD_EXECUTABLE(diffusion_amg_preconditioner diffusion_amg_preconditioner.cc)

TARGET_LINK_LIBRARIES(diffusion_amg_preconditioner ${boomerAMG_solver_lib})
TARGET_INCLUDE_DIRECTORIES(diffusion_amg_preconditioner PRIVATE ${boomerAMG_solver_include})

DEAL_II_SETUP_TARGET(diffusion_amg_preconditioner)

