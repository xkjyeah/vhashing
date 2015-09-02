#include <cuda_runtime.h>
#include <vhashing.h>
#include <MatDevice.h>
#include <MatCV.h>
#include <glog/logging.h>

// NYU dataset camera intrinsics, according to Kinfu
#define WIDTH  640
#define HEIGHT  480
#define fx  518.8579f
#define cx  325.5824f
#define fy  519.4696f
#define cy  253.7362f

#define RESOLUTION 0.01f  // 1cm

struct Voxel {
  float sdf;
  uchar3 color;
  uint8_t weight;
};
struct VoxelBlock {
  Voxel voxels[8*8*8];
};

int3 emptyKey{
  0x7fffffff,
  0x7fffffff,
  0x7fffffff,
};

struct BlockHasher {
	__device__ __host__
	size_t operator()(int3 patch) const {
		const size_t p[] = {
			73856093,
			19349669,
			83492791
		};
		return ((size_t)patch.x / 8 * p[0]) ^
					 ((size_t)patch.y / 8 * p[1]) ^
					 ((size_t)patch.z / 8 * p[2]);
	}
};
struct BlockEqual {
	__device__ __host__
	bool operator()(int3 patch1, int3 patch2) const {
		return patch1.x == patch2.x &&
						patch1.y == patch2.y;
						patch1.z == patch2.z;
	}
};

__device__ __host__
int Mod(int a, int b) {
  return ((a % b) + b) % b;
};
__device__ __host__
int Div(int a, int b) {
  return (a - Mod(a,b)) / b;
};

typedef vhashing::HashTableBase<int3, VoxelBlock, BlockHasher, BlockEqual> HTBase;
typedef vhashing::HashTable<int3, VoxelBlock, BlockHasher, BlockEqual, vhashing::host_memspace> HTHost;
typedef vhashing::HashTable<int3, VoxelBlock, BlockHasher, BlockEqual, vhashing::device_memspace> HTDevice;

struct Unproject {
  __device__ __host__
  int3 operator()(uint16_t depth, int index) {
    float y = 0.5f + index / WIDTH - cy;
    float x = 0.5f + index % WIDTH - cx;

    float metric_depth = depth * 0.001f;

    float3 cam_coord{
      x / fx * metric_depth,
      y / fy * metric_depth,
      metric_depth
    };

    int3 cam_coord_int{
      (int) floor(cam_coord.x / RESOLUTION),
      (int) floor(cam_coord.y / RESOLUTION),
      (int) floor(cam_coord.z / RESOLUTION)
    };

    int3 key_coord{
      cam_coord_int.x - Mod(cam_coord_int.x, 8),
      cam_coord_int.y - Mod(cam_coord_int.y, 8),
      cam_coord_int.z - Mod(cam_coord_int.z, 8)
    };

    return key_coord;
  }
};

__device__ __host__
bool operator< (int3 a, int3 b) {
  return (a.x < b.x) || (a.x == b.x && (
         (a.y < b.y) || (a.y == b.y && (
         (a.z < b.z)))));
}

__device__ __host__
bool operator== (int3 a, int3 b) {
  return (a.x == b.x) &&
         (a.y == b.y) &&
         (a.z == b.z);
}



int main(int argc, char *argv[]) {
  using namespace vhashing;
  google::InitGoogleLogging(argv[0]);

  CHECK(argc >= 2);

  cv::Mat depthmap = cv::imread(argv[1], CV_LOAD_IMAGE_UNCHANGED);

  // allocate blocks...
  MatCV<uint16_t> mcv(depthmap);
  MatDevice<uint16_t> dmcv(mcv);

  CHECK(depthmap.depth() == CV_16U);
  CHECK(dmcv.rows == HEIGHT);
  CHECK(dmcv.cols == WIDTH);

  thrust::device_ptr<uint16_t> begin((uint16_t*)dmcv.data_);
  thrust::device_ptr<uint16_t> end((uint16_t*)dmcv.data_ + WIDTH * HEIGHT);

  thrust::counting_iterator<int> ci(0);

  // convert to camera coordinates...
  thrust::device_vector<int3> key_coordinates(WIDTH * HEIGHT);

  thrust::transform(
      begin, end,
      ci,
      key_coordinates.begin(),
      Unproject());

  // 
  HTDevice htdevice(10000, 5, 20000, emptyKey);
  thrust::sort(key_coordinates.begin(),
                key_coordinates.end());
  auto last_it = thrust::unique(key_coordinates.begin(),
                key_coordinates.end());
  printf("There are %d unique blocks to allocate\n",
      (int)thrust::distance(key_coordinates.begin(), last_it));

  htdevice.AllocKeysNoDups(key_coordinates.begin(), last_it);

  // print out
  HTHost hthost(htdevice);
  hthost.Apply([] (int3 key, VoxelBlock &v) {
      printf("allocated voxel block: %d %d %d\n", key.x, key.y, key.z);
      });
}

