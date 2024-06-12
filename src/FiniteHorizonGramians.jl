module FiniteHorizonGramians

using LinearAlgebra
using PrecompileTools
using SimpleUnPack
using FastBroadcast

include("utils.jl")
include("normtols.jl")

abstract type AbstractExpAndGramAlgorithm end

export AbstractExpAndGramAlgorithm

"""
    exp_and_gram(
        A::AbstractMatrix{T},
        B::AbstractMatrix{T},
        t::Number,
        method::AbstractExpAndGramAlgorithm,
    )

Compute the matrix exponential exp(A) and the controllability Gramian of (A, B) over the interval [0, t],
if t is omitted the unit interval is used.
"""
function exp_and_gram end

"""
    exp_and_gram_chol(
        A::AbstractMatrix{T},
        B::AbstractMatrix{T},
        [t::Number],
        method::AbstractExpAndGramAlgorithm,
    )
Computes the matrix exponential of A * t and the controllability Gramian of (A, B) on the interval [0, t],
if t is omitted the unit interval is used.
"""
function exp_and_gram_chol end


"""
    exp_and_gram!(
        eA::AbstractMatrix{T},
        G::AbstractMatrix{T},
        A::AbstractMatrix{T},
        B::AbstractMatrix{T},
        t::Number,
        method::AbstractExpAndGramAlgorithm,
        cache = alloc_mem(A, B, method),
    )

Computes the matrix exponential of A * t and the controllability Gramian of (A, B) on the interval [0, t],
if t is omitted the unit interval is used.
The result is stored in (eA, G), which are returned.
"""
function exp_and_gram! end

"""
    exp_and_gram_chol!(
        eA::AbstractMatrix{T},
        U::AbstractMatrix{T},
        A::AbstractMatrix{T},
        B::AbstractMatrix{T},
        [t::Number],
        method::ExpAndGram{T,q},
        [cache = alloc_mem(A, B, method)],
    )

    exp_and_gram_chol!(
        eA::AbstractMatrix{T},
        U::AbstractMatrix{T},
        A::AbstractMatrix{T},
        B::AbstractMatrix{T},
        [t::Number],
        method::AdaptiveExpAndGram,
        [cache = nothing],
    )

Computes the matrix exponential of A * t and the controllability Gramian of (A, B) on the interval [0, t],
if t is omitted the unit interval is used.
The result is stored in (eA, U), which are returned.
"""
function exp_and_gram_chol! end

export exp_and_gram, exp_and_gram!, exp_and_gram_chol, exp_and_gram_chol!

include("exp_and_gram.jl")
include("adaptive_exp_and_gram.jl")
export ExpAndGram, AdaptiveExpAndGram

include("precompile.jl")

end
