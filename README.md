# MPI_dot

This repo teaches parallel thinking and introduces MPI in a 40-minute hands-on lesson.

The Message Passing Interface (MPI) is widely used in High Performance Computing (HPC) to distribute work across multiple processors. It enables users to take advantage of **distributed memory parallelism** — that is, using separate pools of memory on different nodes by sending messages between them.

In this model, the programmer is responsible for dividing the work and must plan how to use the libraries and tools provided by MPI to achieve their goal.

Because parallel thinking is essential before writing MPI code, we will start with the **mathematics of the dot product** and show how this simple operation can be easily divided and performed in parallel across multiple processes.

This hands-on lesson walks through:
- Understanding how to split up the dot product across multiple workers (before introducing MPI)
- Writing a basic serial version of the dot product
- Writing a parallel version using `MPI_Scatter` and `MPI_Reduce`
- Scaling up to large arrays that benefit from distributed execution

