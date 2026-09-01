#pragma once

// Annotation shims so host-only compilation of annotation-bearing BCC32
// headers does not require <cuda_runtime.h>. Under nvcc (__CUDACC__) the real
// keywords come from CUDA itself and this header defines nothing.
#if !defined(__CUDACC__)
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#endif
