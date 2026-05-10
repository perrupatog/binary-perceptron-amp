# Binary Perceptron AMP

Julia code for studying the binary perceptron with Approximate Message Passing (AMP), with support for both:

- `:teacher_student` instances, where labels are generated from a planted binary teacher
- `:storage` instances, where labels are sampled at random

The binary output labels are generated through the square-wave activation

```math
\phi(h) = -\mathrm{sign}\!\left(\sin\left(\frac{\pi h}{\delta}\right)\right),
```

implemented in the code as a strictly binary map taking values in `{−1, +1}`.
For large `δ`, this reduces to the standard asymmetric perceptron, since in that
regime the sign is controlled by the sign of the local field `h`.

The main implementation lives in [`src/BinaryPerceptronAMP.jl`](src/BinaryPerceptronAMP.jl).

## What This Repo Does

This module provides three main entry points:

- `FixedPoint`: solve the AMP fixed-point equations without reinforcement
- `FindSolution`: run reinforced AMP to search for a satisfying binary configuration
- `ComputeOptReinforcement`: estimate the largest reinforcement scale for which the algorithm still succeeds

Both `FindSolution` and `ComputeOptReinforcement` support:

- `problem = :teacher_student`
- `problem = :storage`

## Requirements

This project uses Julia and the following packages:

- `SpecialFunctions`
- `Distributions`

Standard-library modules used by the code:

- `Random`
- `Statistics`
- `LinearAlgebra`

A minimal setup is:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

From the shell, the equivalent first-time setup is:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Usage

Load the module from the repository root:

```julia
include("src/BinaryPerceptronAMP.jl")
using .BinaryPerceptronAMP
```

Or run the small public example:

```bash
julia --project=. scripts/example.jl
```

### Fixed Point

```julia
BinaryPerceptronAMP.FixedPoint(1000, 0.5; δ = 1.0, n_samples = 1, n_restarts = 10)
```

This runs the non-reinforced fixed-point iteration on a teacher-student instance and prints summary information for each sample.

### Find a Solution

Teacher-student:

```julia
res = BinaryPerceptronAMP.FindSolution(1000, 0.5, 1e-3; problem = :teacher_student, δ = 1.0)
```

Storage:

```julia
res = BinaryPerceptronAMP.FindSolution(1000, 0.5, 1e-3; problem = :storage, δ = 1.0)
```

`FindSolution` prints one summary line per sample and returns:

- a `NamedTuple` if `n_samples == 1`
- a `Vector{NamedTuple}` if `n_samples > 1`

The returned object contains fields such as `problem`, `α`, `λ`, `n_attempts`, `nSAT`, `sat_fraction`, and `scal_prod` (`nothing` in storage mode).

### Optimize the Reinforcement Scale

Teacher-student:

```julia
res = BinaryPerceptronAMP.ComputeOptReinforcement(1000, 0.5; problem = :teacher_student, δ = 1.0)
```

Storage:

```julia
res = BinaryPerceptronAMP.ComputeOptReinforcement(1000, 0.5; problem = :storage, δ = 1.0)
```

This routine performs a bisection search on the reinforcement scale and returns structured results in the same style as `FindSolution`.

## Notes

- `α` is the constraint density, with `P = round(Int, N * α)`
- `δ` controls the square-wave labeling function used in the code
- output is intentionally lightweight and suited for batch runs or shell redirection

## License

This project is released under the MIT License. See [LICENSE](LICENSE).

## References

[1] Marco Benedetti, Andrej Bogdanov, Enrico M. Malatesta, Marc Mézard, Gianmarco Perrupato, Alon Rosen, Nikolaj I. Schwartzbach, Riccardo Zecchina, *Overlap Gap and Computational Thresholds in the Square Wave Perceptron*, arXiv:2506.05197, 2025. Available at: <https://arxiv.org/abs/2506.05197>

[2] Marco Benedetti, Andrej Bogdanov, Enrico M. Malatesta, Marc Mézard, Gianmarco Perrupato, Alon Rosen, Nikolaj I. Schwartzbach, Riccardo Zecchina, *Are Neural Networks Collision Resistant?*, arXiv:2509.20262, 2025. Available at: <https://arxiv.org/abs/2509.20262>

As of May 10, 2026, `Overlap Gap and Computational Thresholds in the Square Wave Perceptron` is also listed online as published in *Journal of Statistical Mechanics: Theory and Experiment* (2025). As of May 10, 2026, `Are Neural Networks Collision Resistant?` is listed on Alon Rosen's publications page as appearing in *Physical Review X* (2026).
