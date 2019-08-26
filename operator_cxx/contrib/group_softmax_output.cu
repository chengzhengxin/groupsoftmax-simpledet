/*!
 * Copyright (c) 2019 by visi
 * \file group_softmax_output.cu
 * \brief
 * \author zhengxin cheng
*/

#include "./group_softmax_output-inl.h"
#include "../../common/cuda_utils.h"
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <algorithm>
#include <vector>

#define CUDA_1D_KERNEL_LOOP(i, n)                               \
for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
     i += blockDim.x * gridDim.x)

constexpr int CAFFE_CUDA_NUM_THREADS = 512;
constexpr int CAFFE_MAXIMUM_NUM_BLOCKS = 4096;

inline int CAFFE_GET_BLOCKS(const int N) {
  return std::min((N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS,
                  CAFFE_MAXIMUM_NUM_BLOCKS);
}

namespace mshadow {
namespace cuda {

template <typename T>
__global__ void GroupSoftmaxGradKernel(const int nthreads,
                                  T* dstd,
                                  const T* labeld,
                                  const T* groupd,
                                  const int batch_size,
                                  const int label_size,
                                  const int group_step) {
  CUDA_1D_KERNEL_LOOP(idx, nthreads) {
    const T* gd = groupd + idx / group_step * label_size;
    const int l = static_cast<int>(labeld[idx]);
    const T g = gd[l];
    T psum = T(0.0f);
    T* mdstd = dstd + idx * label_size;
    for (int j = 0; j < label_size; ++j) {
      if(g == gd[j])
        psum += mdstd[j];
    }
    psum = (psum - T(1.0f)) / (psum + T(0.00001f));
    for (int j = 0; j < label_size; ++j) {
      if(g == gd[j])
        mdstd[j] *= psum;
    }
  }
}

template<typename DType>
inline void GroupSoftmaxGrad(Tensor<gpu, 2, DType> dst,
                        const Tensor<gpu, 2, DType> &src,
                        const Tensor<gpu, 1, DType> &label,
                        const Tensor<gpu, 2, DType> &group) {
  Copy(dst, src, src.stream_);
  DType *dstd = dst.dptr_;
  const DType *labeld = label.dptr_;
  const DType *groupd = group.dptr_;
  const int batch_size = src.size(0);
  const int label_size = src.size(1);
  const int group_step = batch_size / group.size(0);
  const int count = batch_size;
  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridDim, (gridSize + kMaxGridDim - 1) / kMaxGridDim);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "GroupSoftmaxGrad");
  cudaStream_t stream = Stream<gpu>::GetStream(dst.stream_);
  GroupSoftmaxGradKernel<DType><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
      count, dstd, labeld, groupd, batch_size, label_size, group_step);
}


template <typename T>
__global__ void GroupSoftmaxGradKernel(const int nthreads,
                                  T* dstd,
                                  const T* labeld,
                                  const T* groupd,
                                  const int ignore_label,
                                  const int batch_size,
                                  const int label_size,
                                  const int group_step) {
  CUDA_1D_KERNEL_LOOP(idx, nthreads) {
    T* mdstd = dstd + idx * label_size;
    const int l = static_cast<int>(labeld[idx]);
    if (l == ignore_label) {
      for (int j = 0; j < label_size; ++j) {
        mdstd[j] = T(0.0f);
      }
    } else {
      const T* gd = groupd + idx / group_step * label_size;
      const T g = gd[l];
      T psum = T(0.0f);
      for (int j = 0; j < label_size; ++j) {
        if(g == gd[j])
          psum += mdstd[j];
      }
      psum = (psum - T(1.0f)) / (psum + T(0.00001f));
      for (int j = 0; j < label_size; ++j) {
        if(g == gd[j])
          mdstd[j] *= psum;
      }
    }
  }
}

template<typename DType>
inline void GroupSoftmaxGrad(Tensor<gpu, 2, DType> dst,
                        const Tensor<gpu, 2, DType> &src,
                        const Tensor<gpu, 1, DType> &label,
                        const Tensor<gpu, 2, DType> &group,
                        const DType &ignore_label) {
  Copy(dst, src, src.stream_);
  DType *dstd = dst.dptr_;
  const DType *labeld = label.dptr_;
  const DType *groupd = group.dptr_;
  const int batch_size = src.size(0);
  const int label_size = src.size(1);
  const int group_step = batch_size / group.size(0);
  const int count = batch_size;
  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridDim, (gridSize + kMaxGridDim - 1) / kMaxGridDim);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "GroupSoftmaxGrad");
  cudaStream_t stream = Stream<gpu>::GetStream(dst.stream_);
  GroupSoftmaxGradKernel<DType><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
      count, dstd, labeld, groupd, static_cast<int>(ignore_label), batch_size, label_size, group_step);
}



template <typename T>
__global__ void GroupSoftmaxGrad3DKernel(const int nthreads,
                                  T* dstd,
                                  const T* labeld,
                                  const T* groupd,
                                  const int batch_size,
                                  const int depth_size,
                                  const int label_size,
                                  const int group_step) {
  CUDA_1D_KERNEL_LOOP(idx, nthreads) {
    //3D shape: (n, c, d)
    const int bsi = idx / depth_size;   // n
    const int dsi = idx % depth_size;   // d
    const T* gd = groupd + bsi / group_step * label_size;
    const int l = static_cast<int>(labeld[idx]);
    const T g = gd[l];
    T psum = T(0.0f);
    T* mdstd = dstd + bsi * label_size * depth_size + dsi;
    for (int j = 0; j < label_size; ++j) {
      if(g == gd[j])
        psum += mdstd[j * depth_size];
    }
    psum = (psum - T(1.0f)) / (psum + T(0.00001f));
    for (int j = 0; j < label_size; ++j) {
      if(g == gd[j])
        mdstd[j * depth_size] *= psum;
    }
  }
}

template<typename DType>
inline void GroupSoftmaxGrad(Tensor<gpu, 3, DType> dst,
                        const Tensor<gpu, 3, DType> &src,
                        const Tensor<gpu, 2, DType> &label,
                        const Tensor<gpu, 2, DType> &group) {
  Copy(dst, src, src.stream_);
  DType *dstd = dst.dptr_;
  const DType *labeld = label.dptr_;
  const DType *groupd = group.dptr_;
  const int batch_size = src.size(0);
  const int label_size = src.size(1);
  const int depth_size = src.size(2);
  const int group_step = batch_size / group.size(0);
  const int count = batch_size * depth_size;
  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridDim, (gridSize + kMaxGridDim - 1) / kMaxGridDim);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "GroupSoftmaxGrad");
  cudaStream_t stream = Stream<gpu>::GetStream(dst.stream_);
  GroupSoftmaxGrad3DKernel<DType><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
      count, dstd, labeld, groupd, batch_size, depth_size, label_size, group_step);
}


template <typename T>
__global__ void GroupSoftmaxGrad3DKernel(const int nthreads,
                                  T* dstd,
                                  const T* labeld,
                                  const T* groupd,
                                  const int ignore_label,
                                  const int batch_size,
                                  const int depth_size,
                                  const int label_size,
                                  const int group_step) {
  CUDA_1D_KERNEL_LOOP(idx, nthreads) {
    //3D shape: (n, c, d)
    const int bsi = idx / depth_size;   // n
    const int dsi = idx % depth_size;   // d
    const int l = static_cast<int>(labeld[idx]);
    T* mdstd = dstd + bsi * label_size * depth_size + dsi;
    if (l == ignore_label) {
      for (int j = 0; j < label_size; ++j) {
        mdstd[j * depth_size] = T(0.0f);
      }
    } else {
      const T* gd = groupd + bsi / group_step * label_size;
      const T g = gd[l];
      T psum = T(0.0f);
      for (int j = 0; j < label_size; ++j) {
        if(g == gd[j])
          psum += mdstd[j * depth_size];
      }
      psum = (psum - T(1.0f)) / (psum + T(0.00001f));
      for (int j = 0; j < label_size; ++j) {
        if(g == gd[j])
          mdstd[j * depth_size] *= psum;
      }
    }
  }
}

template<typename DType>
inline void GroupSoftmaxGrad(Tensor<gpu, 3, DType> dst,
                        const Tensor<gpu, 3, DType> &src,
                        const Tensor<gpu, 2, DType> &label,
                        const Tensor<gpu, 2, DType> &group,
                        const DType &ignore_label) {
  Copy(dst, src, src.stream_);
  DType *dstd = dst.dptr_;
  const DType *labeld = label.dptr_;
  const DType *groupd = group.dptr_;
  const int batch_size = src.size(0);
  const int label_size = src.size(1);
  const int depth_size = src.size(2);
  const int group_step = batch_size / group.size(0);
  const int count = batch_size * depth_size;
  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridDim, (gridSize + kMaxGridDim - 1) / kMaxGridDim);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "GroupSoftmaxGrad");
  cudaStream_t stream = Stream<gpu>::GetStream(dst.stream_);
  GroupSoftmaxGrad3DKernel<DType><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
      count, dstd, labeld, groupd, static_cast<int>(ignore_label), batch_size, depth_size, label_size, group_step);
}

}  // namespace cuda

template<typename DType>
inline void GroupSoftmaxGrad(Tensor<gpu, 2, DType> dst,
                        const Tensor<gpu, 2, DType> &src,
                        const Tensor<gpu, 1, DType> &label,
                        const Tensor<gpu, 2, DType> &group) {
  cuda::GroupSoftmaxGrad(dst, src, label, group);
}

template<typename DType>
inline void GroupSoftmaxGrad(Tensor<gpu, 2, DType> dst,
                        const Tensor<gpu, 2, DType> &src,
                        const Tensor<gpu, 1, DType> &label,
                        const Tensor<gpu, 2, DType> &group,
                        const DType &ignore_label) {
  cuda::GroupSoftmaxGrad(dst, src, label, group, ignore_label);
}

template<typename DType>
inline void GroupSoftmaxGrad(Tensor<gpu, 3, DType> dst,
                        const Tensor<gpu, 3, DType> &src,
                        const Tensor<gpu, 2, DType> &label,
                        const Tensor<gpu, 2, DType> &group) {
  cuda::GroupSoftmaxGrad(dst, src, label, group);
}

template<typename DType>
inline void GroupSoftmaxGrad(Tensor<gpu, 3, DType> dst,
                        const Tensor<gpu, 3, DType> &src,
                        const Tensor<gpu, 2, DType> &label,
                        const Tensor<gpu, 2, DType> &group,
                        const DType &ignore_label) {
  cuda::GroupSoftmaxGrad(dst, src, label, group, ignore_label);
}

}  // namespace mshadow

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(GroupSoftmaxOutputParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new GroupSoftmaxOutputOp<gpu, DType>(param);
  })
  return op;
}

}  // namespace op
}  // namespace mxnet

