# Solutions

## Quiz 1

What are 3 traditional ways HW designers make computers run faster?

- [x] Faster clocks
- [ ] Longer clock period
- [x] More work/clock cycle
- [ ] Larger hard disk
- [x] More processors
- [ ] Reduce amount of memory

## Quiz 2

Are processors today getting faster because

- [ ] We're clocking their transistors faster
- [x] We have more transistors available for computing

## Quiz 3

Which techniques are computer designers using today to build more power efficient
chips?

- [ ] Fewer, more complex processors
- [x] More, simpler processors
- [ ] Maximizing the speed of processors
- [ ] Increasing the complexity of the control HW

## Quiz 4

A car and a bus are driving 4 500 km.
The car takes 2 people and drives 200 km/h.
The bus takes 40 people and drives 50 km/h.
What are the latency and throughput?

Car:

- Latency (hours) = 4 500 km / (200 km/h) = 22.5 h
- Throughput (people/hour) = 2 people / 22.5 h = 0.088... p/h

Bus:

- Latency (hours) = 4 500 km / (50 km/h) = 90 h
- Throughput (people/hour) = 40 people / 90 h = 0.444... p/h

## Quiz 5

The GPU can do the following

- [F] Initiate data send GPU -> CPU
- [T] Respond to CPU request to send data GPU -> CPU
- [F] Initiate data send CPU -> GPU
- [T] Respond to CPU request to recv data CPU -> GPU
- [T] Compute a kernel launched by CPU
- [F*] Compute a kernel launched by GPU

F* this was changing as the course were made

## Quiz 6

What is the GPU good at?

- [] Launching a small number of threads efficiently
- [x] Launching a large number of threads efficiently
- [] Running one thread very quickly
- [] Running one thread that does lots of work in parallel
- [x] Running a large number of threads in parallel

## Quiz 7

For the following code

```c
for(int i=0; i<64; ++i){
  out[i] = in[i] * in[i];
}
```

1. How many multiplications?
1. How long to execute if `*` takes 2 ns and everything else is free?

Answers:

1. 64
1. 64 * 2 ns = 128 ns

## Quiz 8

For the same codes as in [quiz 7](#quiz-7):
If we launch 64 threads:

1. How many multiplications?
1. How long to execute if `*` takes 10 ns and everything else is free?

Answers:

1. 64
1. (64 * 10 ns)/ 64 threads = 10 ns

## Quiz 9

In the following code, what should be filled in the `???`?

```c
// transfer the array to the GPU
cudaMemcpy(d_in, h_in, ARRAY_BYTES, ???);

// launch the kernel
cube<<<1, ARRAY_SIZE>>>(d_out, d_in);

// copy back the result array to the CPU
cudaMemcpy(h_out, d_out, ARRAY_BYTES, ???)
```

Answers:

1. `cudaMemcpyHostToDevice`
1. `cudeMemcpyDeviceToHost`

## Quiz 10

```cuda
kernel<<<dim3(8,4,2), dim3(16,16)>>>(...)
```

Questions:

- How many blocks?
- How many threads per block?
- How many total threads?

Answers:

- How many blocks: `8*4*2=64`
- How many threads per block: `16*16=256`
- How many total threads? `64*256=16384`

## Quiz 11

Check the problems that can be solved using `map`

- [] Sort an input array
- [x] Add one to each element in an input array
- [] Sum up all elements in an input array
- [] Compute the average of an input array

Comments:

- Maps are on the form `map(elements, function)`
- Maps have one output per input
- GPUs have many parallel processors
- GPUs optimize for throughput
