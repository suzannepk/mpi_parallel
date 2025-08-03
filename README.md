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


## Step-by-Step: Chunking the Work

Let’s say we have:

- N = 8 total elements to add  
- P = 4 people (lets call these people "processes" because it will help making the transtion to MPI later )

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


# Serial C code for dot product. 

The code below is written in C. In this we do the serial dot prodcut. 

 

```c
#include <stdio.h>           // Include standard input/output header for printf

#define N 1000               // Define a constant N = 1000, size of the arrays

int main() {
    double a[N], b[N], dot = 0.0;  // Declare arrays a and b of size N, and initialize dot to 0.0

    for (int i = 0; i < N; i++) {  // Loop over each index from 0 to N-1
        a[i] = i * 0.5;            // Fill array a with values: a[i] = i * 0.5
        b[i] = i * 2.0;            // Fill array b with values: b[i] = i * 2.0
        dot += a[i] * b[i];        // Accumulate the product of a[i] and b[i] into dot
    }

    printf("Dot product: %f\n", dot);  // Print the final dot product result

    return 0;                    // Exit the program successfully
}

```


Now we will use the same logic that we did when we imagined that we our calss was a human paralle procoessor.


```c
#include <stdio.h>      // For printf
#include <stdlib.h>     // For malloc and free
#include <mpi.h>        // For MPI functions

#define N 10000000      // Total number of elements in each vector

int main(int argc, char** argv) {
    int rank, size;                     // rank = ID of the current process, size = total number of processes
    double *a = NULL, *b = NULL;        // These pointers will hold the full vectors (on rank 0 only)
    double local_dot = 0.0;             // Each process will compute a portion of the dot product
    double global_dot = 0.0;            // Final result of the full dot product (only meaningful on rank 0)

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get the rank of the current process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Get the total number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Calculate how many elements each process should handle
    int chunk = N / size;

    // Allocate local arrays for this process to hold its chunk of data
    double *a_local = malloc(chunk * sizeof(double));
    double *b_local = malloc(chunk * sizeof(double));

    // Only rank 0 initializes the full vectors
   // We use malloc to get access to the heap memory, so each MPI process can allocate a chunk of data at runtime,
   // since stack arrays are limited in size and not suitable for dynamic, distributed work.
    if (rank == 0) {
        a = malloc(N * sizeof(double));   // Allocate memory for vector a
        b = malloc(N * sizeof(double));   // Allocate memory for vector b
        for (int i = 0; i < N; i++) {
            a[i] = i * 0.5;               // Fill vector a with sample values
            b[i] = i * 2.0;               // Fill vector b with sample values
        }
    }

    // Distribute parts of vector a from rank 0 to all processes
    MPI_Scatter(a, chunk, MPI_DOUBLE,    // Send chunk elements from a
                a_local, chunk, MPI_DOUBLE,  // Receive chunk elements into a_local
                0, MPI_COMM_WORLD);      // Root is rank 0

    // Distribute parts of vector b similarly
    MPI_Scatter(b, chunk, MPI_DOUBLE,
                b_local, chunk, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Each process computes the dot product of its own chunk
    for (int i = 0; i < chunk; i++) {
        local_dot += a_local[i] * b_local[i];
    }

    // Reduce all local_dot values into global_dot on rank 0 using a sum operation
    MPI_Reduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Rank 0 prints the result
    if (rank == 0) {
        printf("Global dot product: %f\n", global_dot);
        free(a);   // Free full vectors on rank 0
        free(b);
    }

    // All processes free their local memory
    free(a_local);
    free(b_local);

    // Finalize the MPI environment
    MPI_Finalize();

    return 0;   // Exit the program
}

```









