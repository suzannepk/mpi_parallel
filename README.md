# MPI Intro

This repo teaches parallel thinking and introduces MPI in a 40-minute hands-on lesson.

The Message Passing Interface (MPI) is widely used in High Performance Computing (HPC) to distribute work across multiple processors. It enables users to take advantage of **distributed memory parallelism** — that is, using separate pools of memory on different nodes by sending messages between them.

In this model, the programmer is responsible for dividing the work and must plan how to use the libraries and tools provided by MPI to achieve their goal.

Because parallel thinking is essential before writing MPI code, we will start with the **mathematics of the dot product** and show how this simple operation can be easily divided and performed in parallel across multiple processes.

This hands-on lesson walks through:
- Understanding how to split up the dot product across multiple workers (before introducing MPI)
- Writing a basic serial version of the dot product
- Writing a parallel version using `MPI_Scatter` and `MPI_Reduce`
- Scaling up to large arrays that benefit from distributed execution

# Parallelizing the Dot Product: The Math of Chunking

## Goal

We want to compute the dot product of two vectors `a` and `b` of size `N`:

dot = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + ... + a[N-1]*b[N-1]

Let's imagine that our class is going to divide the work of calculating the elements of this dot product between its members. We are going to give each person a "chunk" of the dot product to calculate so most of the work can be done in parallel, and then one person will be responsible for gathering all the different chunks and adding their results together.

---

## Step-by-Step: Chunking the Work

Let’s say we have:

- `N = 8` total elements to add  
- `P = 4` people (lets call these people "processes" because it will help making the transtion to MPI later )

To split the work evenly:

chunk_size = N / P = 8 / 4 = 2

Each process will have two elements to calculate. 

We'll organize our processes into ranks to make it easier for the gatherer to track them

## Work Division

| Rank (Process) | Index Range | Operation                       |
|----------------|-------------|----------------------------------|
| Rank 0         | 0–1         | `a[0]*b[0] + a[1]*b[1]`          |
| Rank 1         | 2–3         | `a[2]*b[2] + a[3]*b[3]`          |
| Rank 2         | 4–5         | `a[4]*b[4] + a[5]*b[5]`          |
| Rank 3         | 6–7         | `a[6]*b[6] + a[7]*b[7]`          |

Each process computes its own **local dot product**, and then we combine all the local results:

global_dot = local_Rank0 + local_Rank1 + local_Rank2 + local_Rank3

## Example With Real Values

Let:

```c
a = [1, 2, 3, 4, 5, 6, 7, 8]
b = [8, 7, 6, 5, 4, 3, 2, 1]


Total dot product:
  = 1*8 + 2*7 + 3*6 + 4*5 + 5*4 + 6*3 + 7*2 + 8*1
  = 120

| Rank | Elements       | Calculation      | Result |
|------|----------------|------------------|--------|
| 0    | a[0], a[1]     | 1×8 + 2×7        | 22     |
| 1    | a[2], a[3]     | 3×6 + 4×5        | 38     |
| 2    | a[4], a[5]     | 5×4 + 6×3        | 38     |
| 3    | a[6], a[7]     | 7×2 + 8×1        | 22     |

global_dot = 22 + 38 + 38 + 22 = 120


# Serial C code for dot prodcut. 

```










