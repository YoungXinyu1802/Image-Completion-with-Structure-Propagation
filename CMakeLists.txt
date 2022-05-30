cmake_minimum_required(VERSION 3.13)
project(ImageCompletion)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(E:\\eigen\\eigen-3.4.0\\eigen-3.4.0)

add_executable(ImageCompletion
                main.cpp
                StructurePropagation.cpp StructurePropagation.h
                OpenCvUtility.cpp OpenCvUtility.h
                math_function.cpp math_function.h
#                Photometric_Correction.cpp Photometric_Correction.h
#                TextureCompletion.cpp TextureCompletion.h
                AnchorPoint.h)

target_link_libraries(ImageCompletion ${OpenCV_LIBS})