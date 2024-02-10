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
