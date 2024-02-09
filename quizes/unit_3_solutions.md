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
