# Unit 3 solutions

## Quiz 1

Given this tree, where each `o` is an operation

```text
o   o   o   o   o   o   o   o
 \ /     \ /     \ /     \ /
  o       o       o       o
  |       |       |       |
  o       o       o       o
  \      /         \      /
     o                o
     |                |
     o                o
     \                /
      ------   -------
            \ /
             o
```

What is the

- [6] Step complexity
- [21] Work complexity

## Quiz 2

Check the operators that are both binary and associative

- [x] Multiply (`a * b`)
- [x] Minimum (`a min b`)
- [ ] Factorial (`a!`)
- [ ] Logical or (`a || b`)
- [x] Bitwise and (`a & b`)
- [ ] Exponentiation (`a ^ b`)
- [ ] Division (`a / b`)

Comments:

- `(a*b)*c = a*(b*c)`
- `min(min(a, b), c) = min(a, min(b, c))` - always select the min
- Factorial is not binary
- `(a || b) || c = a || (b || c)` - If one is `true` the result is `true`
- `(a & b) & c = a & (b & c)` - Bits need to be simultaneously `1` for the
  result to be `1`
- `4^(3^2) != (4^3)^2`
- `8/(4/2) != (8/4)/2`

## Quiz 3

Which statements are true about a serial reduce code running an input size of
size `n`?

- [ ] It takes `n` operations
- [x] It takes `n-1` operations
- [x] Its work complexity is `O(n)`
- [ ] Its step complexity is `O(1)`

Comment:

The step complexity is also `O(n)`

## Quiz 4

How do you rewrite `((a+b)+c)+d` to allow parallel execution?

Answer:
`(a+b)+(c+d)`

## Quiz 5

What is the step complexity of parallel reduction

- [ ] `sqrt(n)`
- [ ] `n`
- [x] `log2(n)`
- [ ] `n*log2(n)`

Comment:

This is assuming we have enough processors to our disposal, and can be seen by
drawing the tree like so

```text
o   o   o   o   o   o   o   o  ...
 \ /     \ /     \ /     \ /
  o       o       o       o
  |       |       |       |
  \      /         \      /
     o                o
     |                |
     \                /
      ------   -------
            \ /
             o
             |
            ...
```

If there are fewer processor, we can use
[Brent's theorem](https://link.springer.com/referenceworkentry/10.1007/978-0-387-09766-4_80)

```text
T_P <= T + ((N-T)/P)
```

where:

- `T` is the time steps it will take if one had enough processors
- `N` is the number of operations
- `P` is the number of processors

## Quiz 6

In [`3_reduction.cu`](../snippets/3_reduction.cu):
How much more bandwidth does the global memory approach use as compared to the
shared memory approach?

Answer:

3 times

For the global approach we have the following traffic:

| Read | Write |
|------|-------|
| 1024 | 512   |
| 512  | 256   |
| 256  | 128   |
| ...  | ...   |
| 1    | 1     |

Summing up (`sum_0^inf(a/(2^n))=2a`), we find that for reducing `n` values we
need `2n` reads and `n` writes.

For the shared memory, we have the following traffic:

| Read | Write |
|------|-------|
| 1024 | 1     |

Summing up we find that for reducing `n` values we need `n` reads and `1` writes.

## Quiz 7

What are is the identity for the

1. Multiply operator
1. The logical `OR` operator
1. The logical `AND` operator

Answer:

1. `1`, since `a*1=a`
1. `false` since `a || false = a`
1. `true` since `a && true = a`

## Quiz 8

1. What is the identity of max-scan on unsigned ints
1. The output of the max-scan on unsigned ints of the input
  `[3, 1, 4, 1, 5, 9]`?

Answer:

1. `0`
1. `[0, 3, 3, 4, 4, 5]`

## Quiz 9

Turn this serial implementation of inclusive scan

```cpp
int acc = identity;
for(int i=0; i < elements.length(); ++i){
  acc = op(acc, element[i]);
  out[i] = acc;
}
```

into exclusive scan

Answer:

```cpp
int acc = identity;
for(int i=0; i < elements.length(); ++i){
  out[i] = acc;
  acc = op(acc, element[i]);
}
```

into an inclusive scan

## Quiz 10

There is an algorithm whereby the parallel inclusive scan can be found by
performing successive reduction of each element

What is the step complexity of this algorithm

- [ ] `O(1)`
- [x] `O(log n)`
- [ ] `O(n)`
- [ ] `O(n^2)`

What is the work complexity of this algorithm

- [ ] `O(1)`
- [ ] `O(log n)`
- [ ] `O(n)`
- [x] `O(n^2)`

Comments:

- The order of the step complexity is determined by the largest parallel reduce
  we need to perform.
  The step complexity of performing reduction of `n` elements is `n`, hence
  the largest complexity becomes `O(log n)`
- The work complexity can be found in by looking at all the reductions we need
  to do.
  First we need to do reduction for 1 element, then for 2, then for 3, as
  `sum_{0}^{n-1} i` is approximately `n^2/2`, we end up with `O(n^2)`
