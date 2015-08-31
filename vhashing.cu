#include "vhashing.h"
#include <iostream>

struct Point {
  int x, y;
};

struct PointHash {
  __device__ __host__
  uint32_t operator() (Point p) const {
    return p.x ^ p.y;
  }
};

struct PointEqual {
  __device__ __host__
  uint32_t operator() (Point a, Point b) const {
    return a.x == b.x && a.y == b.y;
  }
};

int main() {

  vhashing::HashTable<Point, int, PointHash, PointEqual,
    vhashing::host_memspace> vh(
        100,
        5,
        100,
        Point{1000000,1000000}
        );


  vh[ Point{20,30} ] = 3;
  std::cout << vh[ Point{20,30} ] << std::endl;
}
