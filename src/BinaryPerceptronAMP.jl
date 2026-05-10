#
# BinaryPerceptronAMP.jl
# Approximate Message Passing algorithms for the binary perceptron problem.
# Copyright (c) 2026 Gianmarco Perrupato (perrupatog)
# SPDX-License-Identifier: MIT
#

module BinaryPerceptronAMP

using SpecialFunctions
using Random, Statistics
using Distributions
using LinearAlgebra

const MAX_MAGNETIZATION = 0.999
const Bound_SW = 20 # cutoff square wave

mutable struct Cavity_Fields
        
    a::Vector{Float64}
    h::Vector{Float64}

    a0::Vector{Float64}
    h0::Vector{Float64}

    ω::Vector{Float64}
    g::Vector{Float64}
    dot_g::Vector{Float64}
    V::Vector{Float64}

    Cavity_Fields(N, P) = new(Vector{Float64}(undef, N), Vector{Float64}(undef, N), Vector{Float64}(undef, N), Vector{Float64}(undef, N), Vector{Float64}(undef, P), Vector{Float64}(undef, P), Vector{Float64}(undef, P), Vector{Float64}(undef, P))

end

mutable struct Data

    y::Vector{Float64}
    A::Matrix{Float64}
    A2::Matrix{Float64}
    
    Data(N, P) = new(Vector{Float64}(undef, P), Matrix{Float64}(undef, N, P), Matrix{Float64}(undef, N, P))

end

function Project_Hypercube(field)

    if abs(field) > 0.

        return sign(field)

    else 

        return sign( 2. * rand() - 1. )

    end
    
end

function truncated_tanh(x::Float64)

    return clamp(tanh(x), -MAX_MAGNETIZATION, MAX_MAGNETIZATION)

end

function g_filters(ω, y, V, g0, dot_g0, δ)

    num = 0.
    den = 0.
    ∂num = 0.

    for n in 0:Bound_SW

        si = sin( pi / δ * ( 2 * n + 1 ) * ω )
        co = cos( pi / δ * ( 2 * n + 1 ) * ω )
        ep = exp(- pi^2 * V / δ^2 * ( 2 * n + 1 )^2 / 2. )

        num += ep * co
        den += ep / ( 2 * n + 1 ) * si 
        ∂num += ep * si * ( 2 * n + 1 )

    end

    den = 1. - y * 4 / pi * den
    
    if den > 0

        num = - 4 * y / δ * num 
        ∂num = y * 4 * pi / δ^2 * ∂num

        g = num / den 
        dot_g = - g^2 + ∂num / den

        return g, dot_g
        
    else

        return g0, dot_g0

    end

end

function ϕ(h, δ)

    return (sin( pi / δ * h ) >= 0.) ? -1. : 1.

end

function DatasetGeneration(data::Data, N::Int, P::Int)
        
    for i in 1:P

        xμ = randn(N) / sqrt(N)
        data.A[:, i] = xμ
        data.A2[:, i] = xμ .* xμ

    end
    
end

function ValidateProblem(problem)

    if problem ∉ (:teacher_student, :storage)
        throw(ArgumentError("problem must be :teacher_student or :storage"))
    end

end

function GenerateInstance!(data::Data, wt, N::Int, P::Int, problem, δ)

    DatasetGeneration(data, N, P)

    if problem == :teacher_student
        wt .= [rand(Bool) ? 1 : -1 for _ in 1:N]
        data.y .= ϕ.((wt' * data.A)', δ)
    else
        data.y .= [rand(Bool) ? 1 : -1 for _ in 1:P]
    end

end

function ReinforcedSearch!(cavity_fields, data, w, ξ, wt, N, P, n_restarts, Tmax, λ, δ, problem)

    nSAT = 0
    scal_prod = 0.
    found = false
    n_attempts = 0
    n_iterazioni = 0

    for restart in 1:n_restarts

        Initialize!(cavity_fields, N, P)

        found = false
        iter = 0

        while !found && iter < Tmax

            iter += 1
            λt = λ * iter
            Iter_Reinforcement!(cavity_fields, data, λt, δ)

            w .= Project_Hypercube.(cavity_fields.a)
            ξ .= (w' * data.A)'

            nSAT = count(x -> x > 0, ϕ.(ξ, δ) .* data.y)
            found = (nSAT == P)

            (iter%10==0) && println(nSAT/P," ", sum(cavity_fields.a .^2)/N)

        end

        n_attempts += 1
        n_iterazioni = iter

        if found

            if problem == :teacher_student
                scal_prod = dot(w, wt) / N
            end

            break

        end

    end

    return (
        found = found,
        nSAT = nSAT,
        n_attempts = n_attempts,
        n_iterazioni = n_iterazioni,
        scal_prod = problem == :teacher_student ? scal_prod : nothing,
    )

end

function Iter_Reinforcement!(cavity_fields, data, reinf, δ)

    cavity_fields.a0 .= cavity_fields.a
    cavity_fields.h0 .= cavity_fields.h
    cavity_fields.V .= ((1 .- (cavity_fields.a .^2))' * data.A2)'

    cavity_fields.ω .= (cavity_fields.a' * data.A)' .- (cavity_fields.V .* cavity_fields.g)
    res = g_filters.(cavity_fields.ω, data.y, cavity_fields.V, cavity_fields.g, cavity_fields.dot_g, Ref(δ))
    cavity_fields.g .= [tup[1] for tup in res]
    cavity_fields.dot_g .= [tup[2] for tup in res]
    cavity_fields.h .= - ( data.A2 * cavity_fields.dot_g ) .* cavity_fields.a .+ ( data.A * cavity_fields.g ) 

    ## Reinforcement 

    cavity_fields.h .= cavity_fields.h .+ reinf .* cavity_fields.h0
    cavity_fields.a .= truncated_tanh.(cavity_fields.h)

end

function Iter_FixedPoint!(cavity_fields, data, damp, δ)

    cavity_fields.a0 .= cavity_fields.a
    cavity_fields.h0 .= cavity_fields.h
    cavity_fields.V .= ((1 .- (cavity_fields.a .^2))' * data.A2)'

    cavity_fields.ω .= (cavity_fields.a' * data.A)' .- (cavity_fields.V .* cavity_fields.g)
    res = g_filters.(cavity_fields.ω, data.y, cavity_fields.V, cavity_fields.g, cavity_fields.dot_g, Ref(δ))
    cavity_fields.g .= [tup[1] for tup in res]
    cavity_fields.dot_g .= [tup[2] for tup in res]
    cavity_fields.h .= - ( data.A2 * cavity_fields.dot_g ) .* cavity_fields.a .+ ( data.A * cavity_fields.g ) 
    cavity_fields.a .= tanh.(cavity_fields.h)
    cavity_fields.a .= damp .* cavity_fields.a0 .+ (1 - damp) .* cavity_fields.a
    error = maximum( abs.(cavity_fields.a .- cavity_fields.a0) )

    return error        

end

function Initialize!(cavity_fields, N, P)

    cavity_fields.h .= 0.
    cavity_fields.a .= 0.
    cavity_fields.g .= 0.
    cavity_fields.dot_g .= 0.

end

"""
    FixedPoint()
    FixedPoint(N, α; δ = 1., seme = 123, damp = 0., n_samples = 1, n_restarts = 10, ϵ = 1e-5, Tmax = 30)

Solve the AMP fixed-point equations without reinforcement and report
whether the recovered configuration matches the planted teacher.
"""
function FixedPoint()

    println("AMP.FixedPoint(N, α; δ = 1., seme = 123, damp = 0., n_samples = 1, n_restarts = 10, ϵ = 1e-5, Tmax = 30)")

end

function FixedPoint(N, α; δ = 1., seme = 123, damp = 0., n_samples = 1, n_restarts = 10, ϵ = 1e-5, Tmax = 30)

    Random.seed!(seme)

    P = round(Int, N * α)
    ξ = Vector{Float64}(undef, P)
    w = Vector{Float64}(undef, N)
    wt = Vector{Float64}(undef, N)

    cavity_fields = Cavity_Fields(N, P)
    data = Data(N, P)

    for sample in 1:n_samples

        DatasetGeneration(data, N, P)

        wt .= [rand(Bool) ? 1 : -1 for _ in 1:N]
        data.y .= ϕ.( (wt' * data.A)', δ )   

        scal_prod = 0
        found = false
        n_attempts = 0
        n_iterazioni = 0

        for restart in 1:n_restarts

            Initialize!(cavity_fields, N, P)
                
            found = false
            error = 1.0
            iter = 0

            while !found && iter < Tmax && error > ϵ

                iter += 1
                error = Iter_FixedPoint!(cavity_fields, data, damp, δ)
                w .= Project_Hypercube.(cavity_fields.a)
                ξ .= (w' * data.A)'
                scal_prod = count(x -> x > 0, w .* wt )

                (scal_prod == N) && (found = true)

            end

            n_attempts += 1
            n_iterazioni = iter

            found && break 
  
        end

        println(α, " ", δ, " ", n_attempts, " ", n_iterazioni," " , n_samples, " ", scal_prod / N, " ", found)

    end

end

"""
    FindSolution()
    FindSolution(N, α, λ; problem = :teacher_student, δ = 1., seme = 123, n_samples = 1, n_restarts = 2, Tmax = 1e3)

Run reinforced AMP to search for a satisfying binary configuration.
The `problem` keyword switches between planted teacher-student labels
and random-label storage instances.
"""
function FindSolution()

    println("Usage: AMP.FindSolution(N, α, λ; problem = :teacher_student, δ = 1., seme = 123, n_samples = 1, n_restarts = 2, Tmax = 1e3)")
    println("Output (teacher_student): α  δ  scal_prod  n_attempts  n_iterazioni  n_samples  nSAT / P  found")
    println("Output (storage): α  δ  n_attempts  n_iterazioni  n_samples  nSAT / P  found")
    println("Return: NamedTuple if n_samples == 1, otherwise Vector{NamedTuple}")

end

function FindSolution(N, α, λ; problem = :teacher_student, δ = 1., seme = 123, n_samples = 1, n_restarts = 2, Tmax = 1e3)

    ValidateProblem(problem)

    Random.seed!(seme)

    P = round(Int, N * α)
    ξ = Vector{Float64}(undef, P)
    w = Vector{Float64}(undef, N)
    wt = Vector{Float64}(undef, N)

    cavity_fields = Cavity_Fields(N, P)
    data = Data(N, P)
    results = NamedTuple[]

    for sample in 1:n_samples

        GenerateInstance!(data, wt, N, P, problem, δ)
        search = ReinforcedSearch!(cavity_fields, data, w, ξ, wt, N, P, n_restarts, Tmax, λ, δ, problem)

        if problem == :teacher_student
            println(α, " ", δ, " ", search.scal_prod, " ", search.n_attempts, " ", search.n_iterazioni, " ", n_samples, " ", search.nSAT / P, " ", search.found)
        else
            println(α, " ", δ, " ", search.n_attempts, " ", search.n_iterazioni, " ", n_samples, " ", search.nSAT / P, " ", search.found)
        end

        push!(results, (
            problem = problem,
            N = N,
            α = α,
            λ = λ,
            δ = δ,
            sample = sample,
            n_samples = n_samples,
            n_attempts = search.n_attempts,
            n_iterazioni = search.n_iterazioni,
            found = search.found,
            nSAT = search.nSAT,
            P = P,
            sat_fraction = search.nSAT / P,
            scal_prod = search.scal_prod,
        ))

    end

    return n_samples == 1 ? only(results) : results

end

"""
    ComputeOptReinforcement()
    ComputeOptReinforcement(N, α; problem = :teacher_student, δ = 1., seme = 123, n_samples = 1, n_restarts = 1, tol = 1e-8, n_max_bisections = 20, λmin = 1e-5, tmax = 2.)

Estimate the largest reinforcement scale `λ0` for which reinforced AMP
still finds a solution, using a bisection search over `λ`.
"""
function ComputeOptReinforcement()

    println("Usage: AMP.ComputeOptReinforcement(N, α; problem = :teacher_student, δ = 1., seme = 123, n_samples = 1, n_restarts = 1, tol=1e-8, n_max_bisections=20, λmin=nothing, tmax=2.)")
    println("Output (teacher_student): N  α  δ  scal_prod  n_attempts  n_iterazioni  n_samples  success  λ0")
    println("Output (storage): N  α  δ  n_attempts  n_iterazioni  n_samples  success  λ0")
    println("Return: NamedTuple if n_samples == 1, otherwise Vector{NamedTuple}")

end

function ComputeOptReinforcement(N, α; problem = :teacher_student, δ = 1., seme = 123, n_samples = 1, n_restarts = 1, tol=1e-8, n_max_bisections=20, λmin=1e-5, tmax=2.)

    ValidateProblem(problem)

    Random.seed!(seme)

    P = round(Int, N * α)
    ξ = Vector{Float64}(undef, P)
    w = Vector{Float64}(undef, N)
    wt = Vector{Float64}(undef, N)

    cavity_fields = Cavity_Fields(N, P)
    data = Data(N, P)
    results = NamedTuple[]

    for sample in 1:n_samples

        GenerateInstance!(data, wt, N, P, problem, δ)

        n_bisections = 0
        last_search = (found = false, nSAT = 0, n_attempts = 0, n_iterazioni = 0, scal_prod = nothing)
        best_search = nothing

        λ = 1.
        λ1 = 1.
        λ0 = 0.

        while abs(λ1 - λ0) > tol && n_bisections < n_max_bisections && λ1 > λmin

            Tmax = max(30, tmax / λ)
            n_bisections += 1

            search = ReinforcedSearch!(cavity_fields, data, w, ξ, wt, N, P, n_restarts, Tmax, λ, δ, problem)
            last_search = search

            if search.found

                λ0 = λ
                best_search = search

            else

                λ1 = λ

            end

            λ = (λ1 + λ0) / 2.

            (λ0 == 1) && break

        end

        success = !isnothing(best_search)
        scal_prod = success ? best_search.scal_prod : 0.

        if problem == :teacher_student
            println(N, " ", α, " ", δ, " ", scal_prod, " ", last_search.n_attempts, " ", last_search.n_iterazioni, " ", n_samples, " ", success, " ", λ0)
        else
            println(N, " ", α, " ", δ, " ", last_search.n_attempts, " ", last_search.n_iterazioni, " ", n_samples, " ", success, " ", λ0)
        end

        push!(results, (
            problem = problem,
            N = N,
            α = α,
            δ = δ,
            sample = sample,
            n_samples = n_samples,
            n_attempts = last_search.n_attempts,
            n_iterazioni = last_search.n_iterazioni,
            success = success,
            λ0 = λ0,
            λmin = λmin,
            n_bisections = n_bisections,
            nSAT = last_search.nSAT,
            P = P,
            sat_fraction = last_search.nSAT / P,
            scal_prod = problem == :teacher_student ? scal_prod : nothing,
        ))

    end

    return n_samples == 1 ? only(results) : results

end

end # module
