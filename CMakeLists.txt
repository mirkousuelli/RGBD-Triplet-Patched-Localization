cmake_minimum_required(VERSION 3.0.0)
project(main VERSION 0.1.0)

include(CTest)
enable_testing()

find_package( OpenCV REQUIRED )
find_package( PCL 1.3 REQUIRED COMPONENT common io VISUALIZATION )

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${PCL_INCLUDE_DIRS} )

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(main Code/main.cpp)
add_executable(corners Code/corners.cpp)
add_executable(calibration Code/calibration.cpp)
add_executable(point_cloud Code/point_cloud.cpp)

target_link_libraries( main ${OpenCV_LIBS} )
target_link_libraries( corners ${OpenCV_LIBS} )
target_link_libraries( calibration ${OpenCV_LIBS} )
target_link_libraries( point_cloud ${OpenCV_LIBS} )
target_link_libraries( point_cloud ${PCL_LIBRARIES} ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_VISUALIZATION_LIBRARIES})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)