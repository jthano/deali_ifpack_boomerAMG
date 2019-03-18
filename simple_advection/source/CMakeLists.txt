#src/CMakeLists.txt
#
#SET(CMAKE_INCLUDE_CURRENT_DIR ON)

ADD_EXECUTABLE(simple_advection simple_advection.cc)

TARGET_LINK_LIBRARIES(simple_advection ${boomerAMG_solver_lib})
TARGET_INCLUDE_DIRECTORIES(simple_advection PRIVATE ${boomerAMG_solver_include})

DEAL_II_SETUP_TARGET(simple_advection)

