cmake_minimum_required(VERSION 2.8)
project("vhashing")

find_package(CUDA)

LIST(APPEND CMAKE_MODULES_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")


add_subdirectory(tests)



