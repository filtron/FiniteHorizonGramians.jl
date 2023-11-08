"""
    AdaptiveExpAndGram{T,A} <: AbstractExpAndGramAlgorithm

Adaptive algorithm for computing the matrix exponential and an associated Gramian.

Constructor:

AdaptiveExpAndGram{T}()

creates an algorithm with coefficients stored in the numeric type T.

"""
struct AdaptiveExpAndGram{T,A} <: AbstractExpAndGramAlgorithm where {A}
    methods::A
    function AdaptiveExpAndGram{T}() where {T}
        methods = [ExpAndGram{T,q}() for q in (3, 5, 7, 9, 13)]
        #normtols = T[0.00067, 0.021, 0.13, 0.41, 1.57] # normtols for Gramian
        #normtols = T[0.0014, 0.25, 0.95, 2.1, 5.39] # normtols for matrix exponential
        return new{T,typeof(methods)}(methods)
    end
end

alloc_mem(A, B, method::AdaptiveExpAndGram) = nothing

function exp_and_gram_chol!(
    eA::AbstractMatrix,
    U::AbstractMatrix,
    A::AbstractMatrix,
    B::AbstractMatrix,
    t::Real,
    method::AdaptiveExpAndGram,
    cache::Nothing=nothing,
)
    At, Bt = copy(A) * t, copy(B) * sqrt(t)

    n, _ = _dims_if_compatible(At, Bt)

    methods = method.methods
    normAt = opnorm(At, 1)

    if normAt <= methods[1].normtol && n <= 4
        Φ, U = _exp_and_gram_chol_init!(eA, U, A, B, t, methods[1])
    elseif normAt <= methods[2].normtol && n <= 6
        Φ, U = _exp_and_gram_chol_init!(eA, U, A, B, t, methods[2])
    elseif normAt <= methods[3].normtol && n <= 8
        Φ, U = _exp_and_gram_chol_init!(eA, U, A, B, t, methods[3])
    elseif normAt <= methods[4].normtol && n <= 10
        Φ, U = _exp_and_gram_chol_init!(eA, U, A, B, t, methods[4])
    else
        Φ, U = exp_and_gram_chol!(eA, U, A, B, t, methods[5])
    end

    return Φ, U

end
