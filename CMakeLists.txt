cmake_minimum_required(VERSION 3.0.0)
project(main VERSION 0.1.0)

include(CTest)
enable_testing()

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(main Code/main.cpp)
add_executable(getIAC Code/getIAC.cpp)
add_executable(corners Code/corners.cpp)
add_executable(rectify Code/rectify.cpp)
add_executable(calibration Code/calibration.cpp)

target_link_libraries( main ${OpenCV_LIBS} )
target_link_libraries( getIAC ${OpenCV_LIBS} )
target_link_libraries( corners ${OpenCV_LIBS} )
target_link_libraries( rectify ${OpenCV_LIBS} )
target_link_libraries( calibration ${OpenCV_LIBS} )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)