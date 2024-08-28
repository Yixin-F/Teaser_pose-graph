cmake_minimum_required(VERSION 3.0)
project(asc_optimizer)

SET(CMAKE_BUILD_TYPE Debug) # Release, RelWithDebInfo , Debug
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()
MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})


# SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wno-unused-variable -Wno-unused-parameter -Wno-unused-function -Wno-unused-private-field -Wno-unused-label")
# SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall --warn-unused-variable  --warn-unused-parameter  ")
# SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -warn-unused-variable -warn-unused-parameter -warn-unused-function -warn-unused-private-field -Wno-unused-label")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_STANDARD 17) 

# installed in docker already 
find_package(Eigen3 REQUIRED)

find_package(catkin REQUIRED COMPONENTS 
  roscpp
  rospy
  std_msgs
  message_generation
)
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES cylinder_pub_sub
 CATKIN_DEPENDS roscpp rospy std_msgs message_runtime
#  DEPENDS system_lib
)
include_directories(
# includeprefix
  ${catkin_INCLUDE_DIRS}
)


option(VERBOSE "Venom Similator Printer" ON)
option(DEBUG "Venom Similator Debug" ON)
option(Examples_run "Complile Venom Simulator Examples" OFF)
if(VERBOSE)
 message(STATUS "---Open Venom Simulator Printer---")
 add_definitions(-D__VERBOSE__)
endif()
if(DEBUG)
 message(STATUS "---Open Venom Simulator Debug---")
 add_definitions(-D__DEBUG__)
endif()

# installed later if you need
find_package(Ceres REQUIRED)  # required version-1.14
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

INCLUDE_DIRECTORIES(
        ${CMAKE_SOURCE_DIR}
        ${EIGEN3_INCLUDE_DIR}
        ${EIGEN3_INCLUDE_DIR}/eigen3
        ${CERES_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}
)

file(GLOB SOURCEFILES ${PROJECT_SOURCE_DIR}/src/*/*.hpp ${PROJECT_SOURCE_DIR}/src/*/*.cc ${PROJECT_SOURCE_DIR}/src/*/*/*.cc)

add_library(VenomLib SHARED ${SOURCEFILES})
target_link_libraries(VenomLib
        ${EIGEN3_LIBS}
        ${CERES_LIBRARIES}
        ${catkin_LIBRARIES})

#########---simulation interface---###################
#add_executable(asc_reconstruction_optimizer_node test/test_pose_graph.cc)
#target_link_libraries(asc_reconstruction_optimizer_node VenomLib ${CMAKE_THREAD_LIBS_INIT}) 
add_executable(asc_reconstruction_optimizer_node src/main.cc)
target_link_libraries(asc_reconstruction_optimizer_node VenomLib ${CMAKE_THREAD_LIBS_INIT}) 



