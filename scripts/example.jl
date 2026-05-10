include(joinpath(@__DIR__, "..", "src", "BinaryPerceptronAMP.jl"))
using .BinaryPerceptronAMP

# Small teacher-student example for a quick smoke test.
# Tmax should be ≈ 1/λ
res = BinaryPerceptronAMP.FindSolution(1000, 0.4, 1e-4; problem = :storage, δ = 2.0, n_samples = 1, n_restarts = 1, Tmax = 1e4)

println(res)
