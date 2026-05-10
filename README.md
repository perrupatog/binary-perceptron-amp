# Binary Perceptron AMP

Julia code for studying the binary perceptron with Approximate Message Passing (AMP), with support for both:

- `:teacher_student` instances, where labels are generated from a planted binary teacher
- `:storage` instances, where labels are sampled at random

Binary labels are generated using the square-wave activation

```math
\varphi_{\delta}(h)=\mathrm{sgn}\!\left[\left(\frac{h}{2\delta}+\frac12\right)\bmod 1\right],
```

where the modulo operation is centered in the interval
\([-\tfrac12,\tfrac12)\), i.e. \(x \bmod 1\) denotes the unique real
number in \([-\tfrac12,\tfrac12)\) that differs from \(x\) by an integer.

The implementation uses a strictly binary activation taking values in
\(\{-1,+1\}\). In the large-\(\delta\) limit, the model reduces to the
standard asymmetric perceptron, since the sign is then controlled by the
local field \(h\).

The main implementation lives in [`src/BinaryPerceptronAMP.jl`](src/BinaryPerceptronAMP.jl).

---

## What This Repo Does

This module provides three main entry points:

- `FixedPoint`: solve the AMP fixed-point equations without reinforcement
- `FindSolution`: run reinforced AMP to search for a satisfying binary configuration
- `ComputeOptReinforcement`: estimate the largest reinforcement scale for which the algorithm still succeeds

Both `FindSolution` and `ComputeOptReinforcement` support:

- `problem = :teacher_student`
- `problem = :storage`

---

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

---

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

---

## Fixed Point

```julia
BinaryPerceptronAMP.FixedPoint(
    1000,
    0.5;
    δ = 1.0,
    n_samples = 1,
    n_restarts = 10
)
```

This runs the non-reinforced fixed-point iteration on a teacher-student instance and prints summary information for each sample.

Here the main arguments are:

- `N`: number of variables
- `α`: constraint density, with `P = round(Int, α * N)`

---

## Find a Solution

Teacher-student:

```julia
res = BinaryPerceptronAMP.FindSolution(
    1000,
    0.5,
    1e-3;
    problem = :teacher_student,
    δ = 1.0
)
```

Storage:

```julia
res = BinaryPerceptronAMP.FindSolution(
    1000,
    0.5,
    1e-3;
    problem = :storage,
    δ = 1.0
)
```

`FindSolution` prints one summary line per sample and returns:

- a `NamedTuple` if `n_samples == 1`
- a `Vector{NamedTuple}` if `n_samples > 1`

The returned object contains fields such as:

- `problem`
- `α`
- `λ`
- `n_attempts`
- `nSAT`
- `sat_fraction`
- `scal_prod` (`nothing` in storage mode)

---

## Optimize the Reinforcement Scale

Teacher-student:

```julia
res = BinaryPerceptronAMP.ComputeOptReinforcement(
    1000,
    0.5;
    problem = :teacher_student,
    δ = 1.0
)
```

Storage:

```julia
res = BinaryPerceptronAMP.ComputeOptReinforcement(
    1000,
    0.5;
    problem = :storage,
    δ = 1.0
)
```

This routine performs a bisection search on the reinforcement scale and returns structured results in the same style as `FindSolution`.

---

## Notes

- `α` is the constraint density, with `P = round(Int, α * N)`
- `δ` controls the square-wave labeling function
- `λ` is the reinforcement strength used by reinforced AMP
- output is intentionally lightweight and suited for batch runs or shell redirection

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE).

---

## References

[1] Marco Benedetti, Andrej Bogdanov, Enrico M. Malatesta, Marc Mézard, Gianmarco Perrupato, Alon Rosen, Nikolaj I. Schwartzbach, Riccardo Zecchina,  
*Overlap Gap and Computational Thresholds in the Square Wave Perceptron*,  
Journal of Statistical Mechanics: Theory and Experiment (2025), arXiv:2506.05197.  
Available at: <https://arxiv.org/abs/2506.05197>

[2] Marco Benedetti, Andrej Bogdanov, Enrico M. Malatesta, Marc Mézard, Gianmarco Perrupato, Alon Rosen, Nikolaj I. Schwartzbach, Riccardo Zecchina,  
*Are Neural Networks Collision Resistant?*,  
Physical Review X (2026), arXiv:2509.20262.  
Available at: <https://arxiv.org/abs/2509.20262>
