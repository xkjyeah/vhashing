
include_directories(
  /usr/include/eigen3
  ../include
  ..
  )
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11 -G)
function (build_test fn)
  cuda_add_executable(test_${fn}
    ${fn}.cu
    ../src/safecall.cpp
    )
  target_link_libraries(test_${fn}
      ${OpenCV_LIBRARIES}
      ${Boost_LIBRARIES}
      glog
      )
endfunction()

build_test(lock)
build_test(inserts)

