"""
    AdaptiveExpAndGram{T,A} <: AbstractExpAndGramAlgorithm

Adaptive algorithm for computing the matrix exponential and an associated Gramian.

Constructor:

AdaptiveExpAndGram{T}()

creates an algorithm with coefficients stored in the numeric type T.

"""
struct AdaptiveExpAndGram{T,A} <: AbstractExpAndGramAlgorithm where {A}
    methods::A
end

function AdaptiveExpAndGram{T}() where {T}
    methods = [ExpAndGram{T,q}() for q in (3, 5, 7, 9, 13)]
    #normtols = T[0.00067, 0.021, 0.13, 0.41, 1.57] # normtols for Gramian
    #normtols = T[0.0014, 0.25, 0.95, 2.1, 5.39] # normtols for matrix exponential
    return AdaptiveExpAndGram{T,typeof(methods)}(methods)
end

alloc_mem(A, B, method::AdaptiveExpAndGram) = nothing

function exp_and_gram_chol!(
    eA::AbstractMatrix{T},
    _U::AbstractMatrix{T},
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
    t::Number,
    method::AdaptiveExpAndGram,
    cache = nothing,
) where {T<:Number}
    n, _ = _dims_if_compatible(A::AbstractMatrix, B::AbstractMatrix)


    At = A * t
    Bt = B * sqrt(t)

    methods = method.methods
    normA = opnorm(At, 1)

    if normA <= methods[1].normtol && n <= 4
        Φ, U = _exp_and_gram_chol_init!(eA, _U, At, Bt, methods[1])
    elseif normA <= methods[2].normtol && n <= 6
        Φ, U = _exp_and_gram_chol_init!(eA, _U, At, Bt, methods[2])
    elseif normA <= methods[3].normtol && n <= 8
        Φ, U = _exp_and_gram_chol_init!(eA, _U, At, Bt, methods[3])
    elseif normA <= methods[4].normtol && n <= 10
        Φ, U = _exp_and_gram_chol_init!(eA, _U, At, Bt, methods[4])
    else
        Φ, U = exp_and_gram_chol!(eA, _U, A, B, t, methods[5])
    end

    return Φ, U

end
