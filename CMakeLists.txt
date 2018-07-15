cmake_minimum_required(VERSION 2.8.3)
project(fk_with_nn)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  message_generation
)

find_package(Boost REQUIRED COMPONENTS system)

add_service_files(
  FILES
  Fk.srv
  Ik.srv
  )

generate_messages(
  DEPENDENCIES
  std_msgs
  )

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES jsk_2017_10_semi
#  CATKIN_DEPENDS fetcheus jsk_maps pr2eus
#  DEPENDS system_lib
)



###########
## Build ##
###########

include_directories(
  ${catkin_INCLUDE_DIRS}
)

