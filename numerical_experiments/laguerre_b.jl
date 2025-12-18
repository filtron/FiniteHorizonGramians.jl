# activates the script environment and instantiates it
import Pkg
Base.active_project() != joinpath(@__DIR__, "Project.toml") && Pkg.activate(@__DIR__)
haskey(Pkg.project().dependencies, "FiniteHorizonGramians") ||
    Pkg.develop(path = joinpath(@__DIR__, "../"))
isfile(joinpath(@__DIR__, "Manifest.toml")) && Pkg.resolve()
Pkg.instantiate()

using Revise
includet("ExperimentUtilities.jl")
using FiniteHorizonGramians, Random, .ExperimentUtilities
using MakieCore, CairoMakie, TuePlots, LaTeXStrings


using LinearAlgebra, FiniteHorizonGramians

function mf_gram(A, B)
    n, _ = size(B)
    pre_array = [A B*B'; zero(A) -A']
    post_array = exp(pre_array)
    Φ = post_array[1:n, 1:n]
    G = post_array[1:n, (n+1):2n] * Φ'
    G = hermitianpart!(G)
    return G
end

function lyap_gram(A, B)
    eA = exp(A)
    eAB = eA * B
    Q = B * adjoint(B) - eAB * adjoint(eAB)
    G = lyap(A, Q)
    return G
end

function lyap_gramchol(A, B)
    G = lyap_gram(A, B)
    C = cholesky(Symmetric(G), check = false)
    if !issuccess(C)
        C.factors[:, :] .= NaN
    end
    U = C.factors
    U = triu!(U)
    return U
end

function eigen_gramchol(A, B)
    G = lyap_gram(A, B)
    #G = mf_gram(A, B)

    eA = exp(A)
    eAB = eA * B
    Q = B * adjoint(B) - eAB * adjoint(eAB)
    G = lyap(A, Q)


    E = eigen(Hermitian(G))
    clipped_eigs = max.(E.values, zero(eltype(E.values)))
    U = qr(Diagonal(sqrt.(clipped_eigs)) * adjoint(E.vectors)).R
    return U
end

function ss_gramchol(A, B)
    T = promote_type(eltype(A), eltype(B))
    alg = AdaptiveExpAndGram{T}()
    _, U = exp_and_gram_chol(A, B, alg)
    return U
end

function ref_gram(A, B)
    eA = exp(A)
    G = I - eA * adjoint(eA)
end


function error(G, U)
    any(isnan, U) && return NaN
    return opnorm(adjoint(U) * U - G, 2) / opnorm(G, 2)
end

function rank_estimate(U, tol)
    any(isnan, U) && return 0
    return rank(U, atol = tol)
end


tol = eps(Float64) / 2 # all singular values smaller than unit round-off are considered zero

function error_estimate(A, B)
    T = promote_type(eltype(A), eltype(B))
    R = real(T)
    u = eps(R) / R(2) # unit round off
    err = 2 * (u * opnorm(A, 2) - log(1 - u))
    return err
end


tol = eps(Float64) / 2
λs = (0.5, 1.5, 5.0)
ns = (1000, 2000, 3000, 4000)


function main(λs, ns, tol)
    relerrs = zeros(Float64, length(λs), length(ns))
    ranks = zeros(Int64, length(λs), length(ns))
    err_estimates = zeros(Float64, length(λs), length(ns))
    for (i, λ) in pairs(λs)
        for (j, n) in pairs(ns)
            A = λ * (I - 2 * tril(ones(n, n)))
            B = sqrt(2 * λ) * ones(n, 1)
            G = ref_gram(A, B)
            err_estimate = error_estimate(A, B)
            U = ss_gramchol(A, B)
            e = error(G, U)
            r = rank_estimate(U, tol)
            relerrs[i, j] = e
            ranks[i, j] = r
            err_estimates[i, j] = err_estimate
        end
    end
    return relerrs, ranks, err_estimates
end

relerrs, ranks, err_estimates = main(λs, ns, tol)
