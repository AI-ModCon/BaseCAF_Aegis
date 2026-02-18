# Frontier

Frontier is an AMD Instinct MI250X system at the Oak Ridge Leadership Computing Facility, managed with Slurm.

!!! warning
    Slurm support in Aegis is planned but not yet implemented.

When Frontier support is added, Aegis will generate and submit Slurm batch scripts (via `sbatch`) in the same way it currently handles PBS jobs on Aurora. Configuration, weight staging, and instance orchestration will work through the same `aegis submit` interface.
