cmake_minimum_required(VERSION 3.19)
project(LowtranCppExample CXX Fortran)

file(GENERATE OUTPUT .gitignore CONTENT "*")

set(LOWTRAN_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../lowtran/build/src/lowtran")

include_directories("${LOWTRAN_BUILD_DIR}")

add_executable(my_app main.cpp lowtran_c_wrapper.f90)

target_link_libraries(my_app PRIVATE 
    "${LOWTRAN_BUILD_DIR}/liblowtran.a"
    gfortran
)