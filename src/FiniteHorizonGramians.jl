module FiniteHorizonGramians

using LinearAlgebra
using PrecompileTools

include("utils.jl")

abstract type AbstractExpAndGramAlgorithm end

export AbstractExpAndGramAlgorithm

"""
    exp_and_gram(A::AbstractMatrix, B::AbstractMatrix, alg::AbstractExpAndGramAlgorithm)

Compute the matrix exponential exp(A) and the controllability Gramian of (A, B) over the unit interval.
"""
function exp_and_gram end

"""
    exp_and_gram_chol(A::AbstractMatrix, B::AbstractMatrix, alg::AbstractExpAndGramAlgorithm)

Compute the matrix exponential exp(A) and a upper triangular Cholesky factor of the controllability Gramian of (A, B) over the unit interval.
"""
function exp_and_gram_chol end

export exp_and_gram, exp_and_gram_chol

include("exp_and_gram.jl")
include("adaptive_exp_and_gram.jl")
export ExpAndGram, AdaptiveExpAndGram

include("precompile.jl")

end
