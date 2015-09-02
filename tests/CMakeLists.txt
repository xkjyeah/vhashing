
include_directories(
  /usr/include/eigen3
  ../include
  ..
  )
find_package(OpenCV)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11)
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
build_test(voxelblocks)
build_test(erases)
build_test(filter)
build_test(apply)
build_test(bulkalloc)

