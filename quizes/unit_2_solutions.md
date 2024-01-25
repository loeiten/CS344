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
