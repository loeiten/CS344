# Unit 2 solutions

## Quiz 1

Given a list of basketball players

- name
- height
- rank in height (tallest, 2nd tallest, ...)

If we write each player's record into its location in a sorted list, is this:

- [] A map operation
- [] A gather operation
- [x] A scatter operation

Comment:

- This is a scatter operation as the task is to compute where to write the
  output (note though that scatter is usually one-to-many)
- The map operation is reading from and writing to specific data elements
- The transpose re-orders data elements in memory
  (and must have a 1-to-1 correspondence)
  (1-to-1 correspondence between input and output)
- The gather operation gathers input data elements together
  (many-to-one correspondence between input and output)
- A stencil has a several-to-one correspondence
- A reduce has a all-to-one correspondence
- A scan/sort has a all-to-all correspondence

## Quiz 2

How many times will a given input value be read when applying a 2D von Neumann
stencil, a 2D Moore stencil and a 3D von Neumann stencil?

Answer:

Excluding boundaries: The input will be read equal to the number of elements in
the stencil.
E.g.

- A 2D von Neumann stencil has 5 elements, hence 5 reads
- A 2D Moore stencil has 9 elements, hence 9 reads
- A 3D von Neumann stencil has 7 elements, hence 7 reads

## Quiz 3

In the following code

```cpp
float out[N], in[N];
int i = threadIdx.x;
int j = threadIdx.y;

const float pi = 3.1415;
```

label the code snippets by pattern:

```cpp
// A
out[i] = pi * in[i];

// B
out[i + j*128] = in[j + i*128]

if (i % 2){
  // C
  out[i-1] += pi * in[i]; out[i+1] += pi * in[i]

  // D
  out[i] = (in[i] + in[i-1] + in[i+1]) * pi / 3.0f;
}
```

Answer:

- A: Map as the input exactly maps to the output
- B: Transpose as we are re-ordering elements and there is a 1-to-1
  correspondence
- C: Scatter as the thread is computing for itself where it needs to write the
  result (notice this only happens for every `i % 2` element)
- D: Gather at it is reading from multiple places in the input array.
  It's not a stencil operation as it's only happening for every `i % 2` element

## Quiz 4

Select the true statements

- [x] A thread block contains many threads
- [x] An SM may run more than one block
- [ ] A block may run on more than one SM
- [x] All the threads in a thread block may cooperate to solve a subproblem
- [ ] All the threads that run on a given SM may cooperate to solve a
  subproblem

Comments:

- A kernel runs one or more thread blocks
- A thread block runs one or more thread
- The thread may take different code paths
- SM = streaming multiprocessor
- SM contains of simple processors and memory
- The GPU is responsible for allocating blocks to SMs
- By definition a thread block is run on a single SM
- Threads belonging to different thread blocks on the same SM should not
  cooperate

## Quiz 5

The (...) is responsible for defining the thread blocks in software

- [x] Programmer
- [ ] GPU

The (...) is responsible for allocating the thread blocks to hardware streaming

- [ ] Programmer
- [x] GPU

## Quiz 6

Given a single kernel that is launched on many thread blocks including X and Y,
the programmer can specify ...

- [ ] that block X will run at the same time as block Y
- [ ] that block X will run after block Y
- [ ] that block X will run on SM Z

NOTE: All of the above are false as there ar no such guarantees

## Quiz 7

How many different outputs can different run of this program produce

```cpp
#include <stdio.h>

#define NUM_BLOCKS 16
#define BLOCK_WIDTH 1

__global__ void hello()
{
  printf("Hello world! I'm a thread in block %d\n", blockIdx.x);
}

int main(int argc, char **argv){
  // Launch the kernel
  hello<<<NUM_BLOCKS, BLOCK_WIDTH>>>();

  // Force the printf()s to flush
  cudaDeviceSynchronize();

  printf("That's all!\n");
}
```

Answer:

- [ ] 1
- [ ] 16
- [ ] 65 536
- [x] 21 trillion

Comment:

- `16!` is approximately 21 trillion
- CUDA guarantees that all threads in a block run on the same SM at the same
  time
- CUDA guarantees that all blocks in a kernel finish before any blocks from
  the next kernel

## Quiz 8

- [x] All threads from a block can access the same variable in that blocks
  shared memory
- [x] Threads from two different blocks can access the same variable in global
  memory
- [x] Threads from different blocks have their own copy of local variables in
  local memory
- [x] Threads from the same block have their own copy of local variables in
  local memory

Comment:

- All threads have their own local memory
- Threads in a block have access to a shared memory
- All blocks have access to the global memory
- The CPU memory is usually copied to the GPU memory

## Quiz 9

We want to move every element of an array on step to the "left",
(i.e. `array[0] = array[1]`, `array[1]=array[2]`, ...).

How many barriers are needed in the following code snippet to achieve this?

```cpp
int idx = threadIdx.x;
__shared__ int array[128];
array[idx] = threadIdx.x;
if(idx < 127){
  array[idx] = array[idx + 1];
}
```

Answer:

`3`

```cpp
int idx = threadIdx.x;
__shared__ int array[128];
array[idx] = threadIdx.x;  // This is a write op
__syncthreads();  // The values must be valid before proceeding
if(idx < 127){
  int temp = array[idx + 1]; // This read op must finish before we try to write
  __syncthreads();
  array[idx] = temp;  // If temp would not finish we would be in trouble
  __syncthreads();  // Ensures that ops are finished before accessing array
}
```

## Quiz 10

Which of the following code snippets are correct

```cpp
__global__ void foo(...){
  __shared__ int s[1024];
  int i = threadIdx.x;
  ...
  // A
  __syncthreads();
  s[i] = s[i-1];
  __syncthreads();
  // B
  __syncthreads();
  if (i%2){
    s[i] = s[i-1];
  }
  __syncthreads();
  // C
  __syncthreads();
  s[i] = (s[i-1] + s[i] + s[i+1])/3.0;
  printf("s[%d]=%f", i, s[i]);
  __syncthreads();
}
```

- [ ] A
- [x] B
- [ ] C

Comments:

- In A there is contention:
   - When `i=1` then the code will try to write `s[0]` to `s[1]`
   - When `i=2` then the code will try to write `s[1]` to `s[2]`
   - However, there is no guarantee that `s[1]` finishes before `s[2]`
- In B there is no contention of the elements:
   - When `i=1`, then `1%2 = 1`, so `s[0]` is written to `s[1]`
   - When `i=2` nothing happens
   - When `i=3`, then `3%2 = 1`, so `s[2]` is written to `s[3]`
- In C we have the same problem as in A

To fix the contention issues one would do

1. All read operations to `tmp`
1. `__syncthreads()`
1. All write from `tmp` to the new element
1. `__syncthreads()`

## Quiz 11

Rank these operations from fastest (1) to slowest (4)

```cpp
__global__ void foo(float *x, float *y, float *z){
  __shared__ float a, b, c;
  float s, t u;
  ...
  // A
  s = *x;
  // B
  t = s;
  // C
  a = b;
  // D
  *y = *z;
  ...
}
```

1. B
1. C
1. A
1. D

Comment:

NOTE: This is oversimplified as compiler may promote to registers,
rearrange access and so forth

- We want to maximize the arithmetic intensity:
  `arithmetic intensity = math/memory`
   - That is:
      - Maximize compute ops per thread
      - Minimize time spent on memory per thread
        (`local < shared << gloabl << cpu`)
- A: Read global, write to local
- B: Read local, write to local
- C: Read shared, write to shared
- D: Read global, write to global

## Quiz 12

Which statements have coalesced access pattern?

```cpp
__global__ void foo(float *g){
  float a = 3.14;
  int i = threadIdx.x;

  // A
  g[i] = a;
  // B
  g[i*2] = a;
  // C
  a = g[i]
  // D
  a = g[BLOCK_WIDTH/2 + i];
  // E
  g[i] = a * g[BLOCK_WIDTH/2 + i];
  // F
  g[BLOCK_WIDTH-1 - i] = a;
}
```

- [x] A
- [ ] B
- [x] C
- [x] D
- [x] E
- [x] F

Comment:

- Coalesced memory access is good as the GPU will access a large memory chunk
  at once, so other threads can re-use the same memory transaction
- Strided memory access is worse as one need more memory transaction per thread
- Random memory access is the worst as each thread will need its own memory
  transaction
- B in the example above is strided, the rest are coalesced in read and write

## Quiz 13

Using [`2_atomics.cu`](../snippets/2_atomics.cu), rank the following from
fastest (`1`) to slowest

- [2] `10^6` threads incrementing `10^6` elements
- [4] `10^6` threads atomically incrementing `10^6` elements
- [1] (incorrect answers due to contention) `10^6` threads incrementing `100`
  elements
- [3] `10^6` threads atomically incrementing `100` elements
- [5] `10^7` threads atomically incrementing `100` elements

Comment:

- Atomics are operation that is indivisible which is performed as a single,
  uninterruptible unit and is not subjected to interference from other
  operations or threads
- On GPU only certain operations on certain data types are supported
- One could usually work around this using atomic CAS (compare-and-swap)
- It serialized the access to memory, so it will be slower than non-atomic
- Still no ordering constraints using atomics
- Remember that floating-point arithmetic is non-associative
- Number `1` is faster than number `2` as it touches less memory,
  but produces the wrong answer due to contention
- Number `3` is faster than number `4` for the same reason

## Summary

- Communication patters
   - gather, scatter, stencil, transpose
- GPU hardware & programming model
   - SMs, threads, blocks, ordering
   - Synchronization
   - Memory model - local, global, shared, atomics
- Efficient GPU programming
   - Access memory faster
      - Coalescing global memory
      - Use faster memory
   - Avoid thread divergence
      - If/else branching
      - For-loops taking long time
