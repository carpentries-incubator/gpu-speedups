---
title: "CuPy and Numba on the GPU"
teaching: 0
exercises: 0
questions:
- "What is CuPy?"
- "What is Numba?"
objectives:
- "Understand copying to and from the GPU (host/device interaction)"
- "Understand the similarities and differences between numpy and cupy arrays"
- "Understand how speedups are benchmarked"
- "Understand what makes for fast GPU spedup functions"
- "Explore some available `cupy` functions that are like those in `numpy` but are GPU spedup"
- "Get experience with GPU spedup `ufuncs`"
- "Get experience with CUDA device functions, which are only called on the GPU (`numba.cuda.jit`)"
``"
keypoints:
- "CuPy is NumPy, but for the GPU"
- "Data is copied from the CPU (host) to the GPU (device), where it is computed on. After a computation, it need to be copied back to the CPU to be interacted with by `numpy`, etc"
- "`%timeit` can be used to benchmark the runtime of GPU spedup functions"
- "GPU spedup functions are optimized for at least four things: 1. input size 2. compute complexity 3. CPU/GPU copying 4. data type. Concretely, a gpu spedup function can be slow because the input size is too small, the computation is too simple, there is excessive data copying to/from GPU/CPU, and the input types are excessivly large (e.g. np.float64 vs np.float32)"
- "Make GPU spedup ufuncs with `@numba.vectorize(..., target='cuda')`"
- "Make CUDA device functions with `@numba.cuda.jit(device=True)` "
---


{% include links.md %}

