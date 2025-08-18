# Introduction to MPI and Parallel Computing

This 40-minute hands-on lesson introduces parallel thinking and the Message Passing Interface (MPI), a key tool in High Performance Computing (HPC).

You’ll learn how to:
- Break down a simple problem (the dot product) into parts that can run in parallel
- Write a serial version in C
- Turn it into a parallel version using MPI_Scatter and MPI_Reduce

By the end, you’ll see how MPI lets multiple processes work together using distributed memory parallelism—separate memory pools that communicate by sending messages.


# The Dot Product

We want to compute the dot product of two vectors `a` and `b` of size `N`:

dot = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + ... + a[N-1]*b[N-1]

With N=8 and vectors a and b: 

a = [1, 2, 3, 4, 5, 6, 7, 8]


b = [8, 7, 6, 5, 4, 3, 2, 1]


Total dot product:

  = 1*8 + 2*7 + 3*6 + 4*5 + 5*4 + 6*3 + 7*2 + 8*1
  = 120

Here is how that can be implemented in serial C code. 

# Serial C code for dot product


```c
#include <stdio.h>           // Include standard input/output header for printf

#define N 1000               // Define a constant N = 1000, size of the arrays

int main() {
    double a[N], b[N], dot = 0.0;  // Declare arrays a and b of size N on the stack, and initialize dot to 0.0

    for (int i = 0; i < N; i++) {  // Loop over each index from 0 to N-1
        a[i] = i * 0.5;            // Fill array a with values: a[i] = i * 0.5
        b[i] = i * 2.0;            // Fill array b with values: b[i] = i * 2.0
        dot += a[i] * b[i];        // Accumulate the product of a[i] and b[i] into dot
    }

    printf("Dot product: %f\n", dot);  // Print the final dot product result

    return 0;                    // Exit the program successfully
}

```

## Note on Serial and Parallel Execution  

- The loop above runs in serial, one iteration after the other.  
- However, the values computed in different iterations of the loop do not depend on each other.  
- This creates an easy opportunity to execute the iterations of the loop in parallel on different processors.  

---

## Memory Allocation in C  

C code allocates memory in two main ways: the stack and the heap.  
These are not physical locations in memory, but two different ways the C compiler and runtime manage memory.  

- **Stack**  
  - Fast, automatically managed memory for local variables.  
  - Size must be known at compile time.  
  - Memory disappears when the function ends.  
  - In the serial code above, the entire set of vectors is placed in a single pool of stack memory.  
  - Example stack allocation:  
   
    `double a[N], b[N];`
    

- **Heap**  
  - Flexible, manually managed memory for data that can change size or outlive a single function.  
  - Allocated at runtime using `malloc` and freed with `free`.  
  - We will use `malloc` in the parallel version of this code below.  

# Parallel Thinking

- Imagine our class is dividing the work of calculating a dot product.  
- Each person gets a "chunk" of the dot product to calculate, so most of the work can be done in parallel.  
- One person is then responsible for gathering all the chunks and adding the results together.  

Let’s say we have:  

- **N = 8** total elements to add  
- **P = 4** people (we’ll call these people processes since it helps with the transition to MPI later)  

To split the work evenly:  
'chunk_size = N / P = 8 / 4 = 2'

Each process will handle 2 elements of the dot product.  

We also organize our processes into ranks (process IDs) to make it easier for the gatherer to track them.  

| Rank (Process) | Global Index range |       Operation                  |
|----------------|--------------------|----------------------------------|
| Rank 1         | 0–1                | `a[0]*b[0] + a[1]*b[1]`          |
| Rank 2         | 2–3                | `a[2]*b[2] + a[3]*b[3]`          |
| Rank 3         | 4–5                | `a[4]*b[4] + a[5]*b[5]`          |
| Rank 4         | 6–7                | `a[6]*b[6] + a[7]*b[7]`          |

Each process computes its own **local dot product**, and then we combine all the local results into a single global value:


global_dot = local_Rank0 + local_Rank1 + local_Rank2 + local_Rank3

## Example With Real Values


| Rank | Elements            | Calculation  | Result |
|------|---------------------|--------------|--------|
| 1    | a[0]b[0] + a[1]b[0] | 1×8 + 2×7    | 22     |
| 2    | a[2]b[2] + a[3]b[3] | 3×6 + 4×5    | 38     |
| 3    | a[4]b[4] + a[5]b[5] | 5×4 + 6×3    | 38     |
| 4    | a[6]b[6] + a[7]b[7] | 7×2 + 8×1    | 22     |

global_dot = 22 + 38 + 38 + 22 = 120

# Thinking About Local and Global Arrays

MPI is **distributed-memory parallelism**, which means each process works in its own pool of memory.  

We *could* give every process a full copy of the arrays, but this wastes memory and requires unnecessary data transfers. Instead, MPI distributes only the needed chunks of the global arrays to each process.  

To make this work:  
- Each process has its own local arrays and local indices.  
- The global indices are mapped to local ones, so each process only handles its part of the data.  

## Work Division for Dot Product

| Index Range (Global) | Rank | Local Index | Operation (Global Index)  | Operation (Local Index)                             | Calculation  | Result |   
|----------------------|------|-------------|---------------------------|-----------------------------------------------------|--------------|--------|
| 0–1                  | 0    | 0, 1        | `a[0]b[0] + a[1]b[1]`     | `a_local[0] * b_local[0] + a_local[1] * b_local[1]` | 1×8 + 2×7    | 22     |
| 2–3                  | 1    | 0, 1        | `a[2]b[2] + a[3]b[3]`     | `a_local[0] * b_local[0] + a_local[1] * b_local[1]` | 3×6 + 4×5    | 38     |
| 4–5                  | 2    | 0, 1        | `a[4]b[4] + a[5]b[5]`     | `a_local[0] * b_local[0] + a_local[1] * b_local[1]` | 5×4 + 6×3    | 38     |
| 6–7                  | 3    | 0, 1        | `a[6]b[6] + a[7]b[7]`     | `a_local[0] * b_local[0] + a_local[1] * b_local[1]` | 7×2 + 8×1    | 22     |

---

## Memory in Practice

For a typical MPI program, the number of ranks (processes) is set by the programmer in the run command, not in the code. This makes it easy to experiment with different numbers of processes without modifying the source.  

Each process gets its own separate allocation of memory.  

This memory is usually allocated on the heap*(using `malloc`). Heap allocation allows memory size to be determined at run time, so it can adjust automatically to the number of processes and the chunk size assigned to each.  


# Initializing MPI  

The first thing MPI does when it is initialized is set up a communicator called `MPI_COMM_WORLD`.  

You can think of a communicator as a package that holds all the organizational information for its MPI region in the code. Inside the communicator:  
- Each process is given a rank (its unique ID).  
- The size of the communicator is equal to the total number of ranks.  

The part of the code that will be executed in parallel with an MPI communicator is called the MPI Region.  
It is always sandwiched between calls to `MPI_Init` and `MPI_Finalize`.  

All MPI function calls within the same MPI region get each process’s rank from the communicator.  
The programmer must use logic based on the rank ID to determine which code path each process follows.  


## MPI Scatter and Gather 
The way we will implement the parallel dot product in the code below also uses two more MPI fuctions: MPI_Sactter and MPI_Gather.  

- **MPI_Scatter**: Splits a large dataset into smaller chunks and sends one chunk to each process.  
  - Example: If you have 8 elements and 4 processes, each process gets 2 elements.  

- **MPI_Gather**: Collects data from all processes and assembles it back into a single dataset on the root process.  
  - Example: Each process computes a partial result, and `MPI_Gather` collects all of them into one array at the root.  

Below is the code. Please read the comments in the code. There are other ways to implement this with MPI, but we have chosen this one to be as close to the parallel thinking example as possible. 



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

    // Setup to the MPI Communicator and get the rank of the current process and the total number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
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

    // Each process computes the dot product of its own local chunck using the local indicies, 
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

```
#include <stdio.h>           // Include standard input/output header for printf

#define N 1000               // Define a constant N = 1000, size of the arrays

int main() {
    double a[N], b[N], c[N]; // Declare arrays a, b, and c of size N on the stack

    for (int i = 0; i < N; i++) {   // Loop over each index from 0 to N-1
        a[i] = i * 0.5;             // Fill array a with values: a[i] = i * 0.5
        b[i] = i * 2.0;             // Fill array b with values: b[i] = i * 2.0
        c[i] = a[i] + b[i];         // Compute element-wise addition: c[i] = a[i] + b[i]
    }

    printf("Vector addition (first 5 results):\n");  
    for (int i = 0; i < 5; i++) {   // Print just the first 5 results for readability
        printf("c[%d] = %f\n", i, c[i]);
    }

    return 0;   // Exit the program successfully
}
````

You will need to: 

- Initialize MPI
- Calculate a chunk size based on N. 
- Create local a,b and c variables 
- scatter chuncks of a and b differnt ranks ( there are also clever ways of solving this with using only index math) To

**Suzanne To Do**: Guide them in to writing the parallel code. Explain that this is only one way to slove the paralleliszation 

```
#include <stdio.h>      // For printf
#include <stdlib.h>     // For malloc and free
#include <mpi.h>        // For MPI functions

#define N 10      // Total number of elements in each vector

int main(int argc, char** argv) {
    int rank, size;                     // rank = ID of the current process, size = total number of processes
    double *a = NULL, *b = NULL, *c = NULL;  // Full vectors (only meaningful on rank 0)

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // TODO Setup the MPI Communicator and get the rank and size
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // TODO Calculate how many elements each process should handle
    int chunk = N / size;

    // Allocate local arrays for this process
    double *a_local = malloc(chunk * sizeof(double));
    double *b_local = malloc(chunk * sizeof(double));
    double *c_local = malloc(chunk * sizeof(double));

    // Only rank 0 initializes the full vectors
    if (rank == 0) {
        a = malloc(N * sizeof(double));
        b = malloc(N * sizeof(double));
        c = malloc(N * sizeof(double));   // Allocate memory for the result vector
        for (int i = 0; i < N; i++) {
            a[i] = i * 0.5;
            b[i] = i * 2.0;
        }
    }

    // Scatter chunks of a and b to all processes
    MPI_Scatter(a, chunk, MPI_DOUBLE, a_local, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(b, chunk, MPI_DOUBLE, b_local, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // TODO Each process computes its local vector addition This will use c_local,a_local and b_local. 
    for (int i = 0; i < chunk; i++) {
        c_local[i] = a_local[i] + b_local[i];
    }

    // TODO Gather all chunks of c back to root
    MPI_Gather(c_local, chunk, MPI_DOUBLE, c, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Rank 0 prints the first few results
    if (rank == 0) {
        printf("Vector addition results (first 10 elements):\n");
        for (int i = 0; i < 10; i++) {
            printf("a[%d]=%6.2f, b[%d]=%6.2f, c[%d]=%6.2f\n",
                   i, a[i], i, b[i], i, c[i]);
        }
        free(a);
        free(b);
        free(c);
    }

    // All processes free their local memory
    free(a_local);
    free(b_local);
    free(c_local);

    MPI_Finalize();

    return 0;
}
```




