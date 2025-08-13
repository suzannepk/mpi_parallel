**# MPI Intro

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

| Rank (Process) | Global Index range |       Operation                  |
|----------------|--------------------|----------------------------------|
| Rank 1         | 0–1                | `a[0]*b[0] + a[1]*b[1]`          |
| Rank 2         | 2–3                | `a[2]*b[2] + a[3]*b[3]`          |
| Rank 3         | 4–5                | `a[4]*b[4] + a[5]*b[5]`          |
| Rank 4         | 6–7                | `a[6]*b[6] + a[7]*b[7]`          |

Each process computes its own **local dot product**, and then we combine all the local results:

global_dot = local_Rank0 + local_Rank1 + local_Rank2 + local_Rank3

## Example With Real Values

Let:


a = [1, 2, 3, 4, 5, 6, 7, 8]


b = [8, 7, 6, 5, 4, 3, 2, 1]


Total dot product:

  = 1*8 + 2*7 + 3*6 + 4*5 + 5*4 + 6*3 + 7*2 + 8*1
  = 120

| Rank | Elements            | Calculation  | Result |
|------|---------------------|--------------|--------|
| 1    | a[0]b[0] + a[1]b[0] | 1×8 + 2×7    | 22     |
| 2    | a[2]b[2] + a[3]b[3] | 3×6 + 4×5    | 38     |
| 3    | a[4]b[4] + a[5]b[5] | 5×4 + 6×3    | 38     |
| 4    | a[6]b[6] + a[7]b[7] | 7×2 + 8×1    | 22     |

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
Now we’ll apply the same logic we used when we imagined our class acting as a human parallel processor.
There’s one more concept we need to introduce: global vs. local indexing in MPI.

In our earlier examples, we thought of the data as being globally available — meaning every processor could, in theory, access every index in the arrays:

```a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]```

However, in MPI, each process only gets its own piece of the data. MPI’s job is to distribute the global arrays into local chunks, so each process only stores and works on its own small part.
We determine how big each process’s chunk will be by the number of Elements, N, divided by the number of processes:

chunk_size = N / P 

In the example we ahave been following this is: 

8 / 4 = 2

So for that example, that means 
## Work Division for Dot Product

 Calculation  | Result |     
--------------|--------|
 1×8 + 2×7    | 22     |
 3×6 + 4×5    | 38     |
 5×4 + 6×3    | 38     |
 7×2 + 8×1    | 22     |

| Index Range (Global) | Rank | Local Index | Operation (Global Index)  | Operation (Local Index)                             | Calculation  | Result |   
|----------------------|------|-------------|---------------------------|-----------------------------------------------------|--------------|--------|
| 0–1                  | 0    | 0, 1        | `a[0]b[0] + a[1]b[1]`     | `a_local[0] * b_local[0] + a_local[1] * b_local[1]` | 1×8 + 2×7    | 22     |
| 2–3                  | 1    | 0, 1        | `a[2]b[2] + a[3]b[3]`     | `a_local[0] * b_local[0] + a_local[1] * b_local[1]` | 3×6 + 4×5    | 38     |
| 4–5                  | 2    | 0, 1        | `a[4]b[4] + a[5]b[5]`     | `a_local[0] * b_local[0] + a_local[1] * b_local[1]` | 5×4 + 6×3    | 38     |
| 6–7                  | 3    | 0, 1        | `a[6]b[6] + a[7]b[7]`     | `a_local[0] * b_local[0] + a_local[1] * b_local[1]` | 7×2 + 8×1    | 22     |


For MPI we inititze the MPI enviroment 



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

    // Calculate how many elements each process should handle N/P
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

    // Each process computes the dot product of its own chunk local chunck using the local indicies 
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

OK Now it is your turn. We are going to do a vector addtion.

Here is the serial code. 

You will need to: 

- Initialize MPI
- Create local a,b and c variables 
- calculate a chunk size based on N. 
- scatter chuncks of a and b differnt ranks 


```
#include <stdio.h>
#include <stdlib.h>

#define N 10

int main() {
    // Allocate arrays
    double *a = malloc(N * sizeof(double));
    double *b = malloc(N * sizeof(double));
    double *c = NULL; // Will hold the gathered result on rank 0

    // Initialize a and b
    for (int i = 0; i < N; i++) {
        a[i] = i * 0.5;
        b[i] = i * 2.0;
    }

    // Print arrays A and B
    printf("Serial Array A: ");
    for (int i = 0; i < N; i++) {
        printf("%f ", a[i]);
    }
    printf("\n");

    printf("Serial Array B: ");
    for (int i = 0; i < N; i++) {
        printf("%f ", b[i]);
    }
    printf("\n");

    // Compute C
    for (int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }

    // Print C
    printf("Array C: ");
    for (int i = 0; i < N; i++) {
        printf("%f ", c[i]);
    }
    printf("\n");

    // Free memory
    free(a);
    free(b);
    free(c);

    return 0;
}

```
Let's break down the math to help you code. 

First 




```
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 10

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    int chunk = N / size;   // Number of elements per process
    int start = rank * chunk;

    // Allocate local arrays
    double *a_local = malloc(chunk * sizeof(double));
    double *b_local = malloc(chunk * sizeof(double));
    double *c_local = malloc(chunk * sizeof(double));

    // Initialize only the chunk this rank is responsible for
    for (int i = 0; i < chunk; i++) {
        a_local[i] = (start + i) * 0.5;  // Same formula as before
        b_local[i] = (start + i) * 2.0;
        c_local[i] = a_local[i] + b_local[i];
    }

    // Allocate full arrays only on rank 0
    double *a = NULL, *b = NULL, *c = NULL;
    if (rank == 0) {
        a = malloc(N * sizeof(double));
        b = malloc(N * sizeof(double));
        c = malloc(N * sizeof(double));
    }

    // Gather all local chunks to rank 0
    MPI_Gather(a_local, chunk, MPI_DOUBLE, a, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(b_local, chunk, MPI_DOUBLE, b, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(c_local, chunk, MPI_DOUBLE, c, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Print all elements on rank 0
    if (rank == 0) {
        printf("Array A: ");
        for (int i = 0; i < N; i++) {
            printf("%f ", a[i]);
        }
        printf("\n");

        printf("Array B: ");
        for (int i = 0; i < N; i++) {
            printf("%f ", b[i]);
        }
        printf("\n");

        printf("Array C: ");
        for (int i = 0; i < N; i++) {
            printf("%f ", c[i]);
        }
        printf("\n");

        free(a);
        free(b);
        free(c);
    }

    free(a_local);
    free(b_local);
    free(c_local);

    MPI_Finalize();
    return 0;
}

```



**
