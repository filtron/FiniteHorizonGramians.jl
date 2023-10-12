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

function exp_and_gram(
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
    method::AdaptiveExpAndGram,
) where {T<:Number}
    Φ, G = exp_and_gram!(copy(A), copy(B), method)
    return Φ, G
end

function exp_and_gram!(
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
    method::AdaptiveExpAndGram,
) where {T<:Number}
    Φ, U = exp_and_gram_chol!(A, B, method)
    G = U' * U
    _symmetrize!(G)
    return Φ, G
end

exp_and_gram_chol(
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
    method::AdaptiveExpAndGram{T},
) where {T<:Number} = exp_and_gram_chol!(copy(A), copy(B), method)

function exp_and_gram_chol!(
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
    method::AdaptiveExpAndGram,
) where {T<:Number}
    n, _ = _dims_if_compatible(A::AbstractMatrix, B::AbstractMatrix)

    methods = method.methods
    normA = opnorm(A, 1)

    if normA <= methods[1].normtol && n <= 4
        Φ, U = _exp_and_gram_chol_init(A, B, methods[1])
    elseif normA <= methods[2].normtol && n <= 6
        Φ, U = _exp_and_gram_chol_init(A, B, methods[2])
    elseif normA <= methods[3].normtol && n <= 8
        Φ, U = _exp_and_gram_chol_init(A, B, methods[3])
    elseif normA <= methods[4].normtol && n <= 10
        Φ, U = _exp_and_gram_chol_init(A, B, methods[4])
    else
        Φ, U = exp_and_gram_chol!(A, B, methods[5])
    end

    return Φ, U

end
