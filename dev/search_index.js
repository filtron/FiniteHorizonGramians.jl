var documenterSearchIndex = {"docs":
[{"location":"public/#Algorithm-types","page":"Public","title":"Algorithm types","text":"","category":"section"},{"location":"public/","page":"Public","title":"Public","text":"AdaptiveExpAndGram{T,A}\nExpAndGram{T,N,A,B,C}","category":"page"},{"location":"public/#FiniteHorizonGramians.AdaptiveExpAndGram","page":"Public","title":"FiniteHorizonGramians.AdaptiveExpAndGram","text":"AdaptiveExpAndGram{T,A} <: AbstractExpAndGramAlgorithm\n\nAdaptive algorithm for computing the matrix exponential and an associated Gramian.\n\nConstructor:\n\nAdaptiveExpAndGram{T}()\n\ncreates an algorithm with coefficients stored in the numeric type T.\n\n\n\n\n\n","category":"type"},{"location":"public/#FiniteHorizonGramians.ExpAndGram","page":"Public","title":"FiniteHorizonGramians.ExpAndGram","text":"ExpAndGram{T,N,A,B,C} <: AbstractExpAndGramAlgorithm\n\nNon-adaptive algorithm of order N for computing the matrix exponential and an associated Gramian.\n\nConstructor:\n\nExpAndGram{T,N}()\n\ncreates an algorithm with coefficients stored in the numeric type T of order N. Current supported values of N are 3, 5, 7, 9, 13.\n\n\n\n\n\n","category":"type"},{"location":"public/#Create-cache","page":"Public","title":"Create cache","text":"","category":"section"},{"location":"public/","page":"Public","title":"Public","text":"FiniteHorizonGramians.alloc_mem(A, B, ::ExpAndGram{T,q}) where {T,q}","category":"page"},{"location":"public/#FiniteHorizonGramians.alloc_mem-Union{Tuple{q}, Tuple{T}, Tuple{Any, Any, ExpAndGram{T, q}}} where {T, q}","page":"Public","title":"FiniteHorizonGramians.alloc_mem","text":"alloc_mem(A, B, ::ExpAndGram{T,q})\n\nComputes a cache for repeated computations with the same input arguments (A, B).\n\n\n\n\n\n","category":"method"},{"location":"public/#Methods","page":"Public","title":"Methods","text":"","category":"section"},{"location":"public/","page":"Public","title":"Public","text":"exp_and_gram\nexp_and_gram_chol\nexp_and_gram!\nexp_and_gram_chol!","category":"page"},{"location":"public/#FiniteHorizonGramians.exp_and_gram","page":"Public","title":"FiniteHorizonGramians.exp_and_gram","text":"exp_and_gram(\n    A::AbstractMatrix{T},\n    B::AbstractMatrix{T},\n    t::Number,\n    method::AbstractExpAndGramAlgorithm,\n)\n\nCompute the matrix exponential exp(A) and the controllability Gramian of (A, B) over the interval [0, t], if t is omitted the unit interval is used.\n\n\n\n\n\n","category":"function"},{"location":"public/#FiniteHorizonGramians.exp_and_gram_chol","page":"Public","title":"FiniteHorizonGramians.exp_and_gram_chol","text":"exp_and_gram_chol(\n    A::AbstractMatrix{T},\n    B::AbstractMatrix{T},\n    [t::Number],\n    method::AbstractExpAndGramAlgorithm,\n)\n\nComputes the matrix exponential of A * t and the controllability Gramian of (A, B) on the interval [0, t], if t is omitted the unit interval is used.\n\n\n\n\n\n","category":"function"},{"location":"public/#FiniteHorizonGramians.exp_and_gram!","page":"Public","title":"FiniteHorizonGramians.exp_and_gram!","text":"exp_and_gram!(\n    eA::AbstractMatrix{T},\n    G::AbstractMatrix{T},\n    A::AbstractMatrix{T},\n    B::AbstractMatrix{T},\n    t::Number,\n    method::AbstractExpAndGramAlgorithm,\n    cache = alloc_mem(A, B, method),\n)\n\nComputes the matrix exponential of A * t and the controllability Gramian of (A, B) on the interval [0, t], if t is omitted the unit interval is used. The result is stored in (eA, G), which are returned.\n\n\n\n\n\n","category":"function"},{"location":"public/#FiniteHorizonGramians.exp_and_gram_chol!","page":"Public","title":"FiniteHorizonGramians.exp_and_gram_chol!","text":"exp_and_gram_chol!(\n    eA::AbstractMatrix{T},\n    U::AbstractMatrix{T},\n    A::AbstractMatrix{T},\n    B::AbstractMatrix{T},\n    [t::Number],\n    method::ExpAndGram{T,q},\n    [cache = alloc_mem(A, B, method)],\n)\n\nexp_and_gram_chol!(\n    eA::AbstractMatrix{T},\n    U::AbstractMatrix{T},\n    A::AbstractMatrix{T},\n    B::AbstractMatrix{T},\n    [t::Number],\n    method::AdaptiveExpAndGram,\n    [cache = nothing],\n)\n\nComputes the matrix exponential of A * t and the controllability Gramian of (A, B) on the interval [0, t], if t is omitted the unit interval is used. The result is stored in (eA, U), which are returned.\n\n\n\n\n\n","category":"function"},{"location":"internal/#Internal-methods","page":"Internal","title":"Internal methods","text":"","category":"section"},{"location":"internal/","page":"Internal","title":"Internal","text":"FiniteHorizonGramians._exp_and_gram_chol_init!(eA::AbstractMatrix{T}, U::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}, method::ExpAndGram{T,q}, cache = nothing) where {T,q}\nFiniteHorizonGramians._exp_and_gram_double!(eA, U, s, cache)\nFiniteHorizonGramians._dims_if_compatible(A::AbstractMatrix, B::AbstractMatrix)\nFiniteHorizonGramians.triu2cholesky_factor!(A::AbstractMatrix{T}) where {T<:Number}\nFiniteHorizonGramians._symmetrize!(A::AbstractMatrix{T}) where {T<:Number}","category":"page"},{"location":"internal/#FiniteHorizonGramians._exp_and_gram_chol_init!-Union{Tuple{q}, Tuple{T}, Tuple{AbstractMatrix{T}, AbstractMatrix{T}, AbstractMatrix{T}, AbstractMatrix{T}, ExpAndGram{T, q}}, Tuple{AbstractMatrix{T}, AbstractMatrix{T}, AbstractMatrix{T}, AbstractMatrix{T}, ExpAndGram{T, q}, Any}} where {T, q}","page":"Internal","title":"FiniteHorizonGramians._exp_and_gram_chol_init!","text":"_exp_and_gram_init(\n    A::AbstractMatrix{T},\n    B::AbstractMatrix{T},\n    method::ExpAndGram{T,q},\n    [cache=alloc_mem(A, B, method)],\n    )\n\nComputes the matrix exponential exp(A) and the controllability Grammian\n\n_0^1 e^A t B B e^A*t dt\n\nusing a Legendre expansion of the matrix exponential of order q.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FiniteHorizonGramians._exp_and_gram_double!-NTuple{4, Any}","page":"Internal","title":"FiniteHorizonGramians._exp_and_gram_double!","text":"_exp_and_gram_double!(eA, U, s, cache)\n\nComputes s iterations of the doubling recursions:\n\nΦ_k+1 = Φ_k^2\n\nand\n\nU_k+1 U_k+1 = Φ_k U_k U_k Φ_k + U_k U_k\n\n\n\n\n\n","category":"method"},{"location":"internal/#FiniteHorizonGramians._dims_if_compatible-Tuple{AbstractMatrix, AbstractMatrix}","page":"Internal","title":"FiniteHorizonGramians._dims_if_compatible","text":"_dims_if_compatible(A::AbstractMatrix, B::AbstractMatrix)\n\nThrows DimensionMismatch if A is not square or if the number of rows of B does not equal the number of columns of A. Is equivalent to size(B) if no error is thrown.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FiniteHorizonGramians.triu2cholesky_factor!-Union{Tuple{AbstractMatrix{T}}, Tuple{T}} where T<:Number","page":"Internal","title":"FiniteHorizonGramians.triu2cholesky_factor!","text":"triu2cholesky_factor!(A::AbstractMatrix{T})\n\nIf A is an upper triangular matrix, it computes the product QA in-place, where Q is a unitary transform such that QA is a valid Cholesky factor. If A is not an upper triangular matrix, returns garbage.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FiniteHorizonGramians._symmetrize!-Union{Tuple{AbstractMatrix{T}}, Tuple{T}} where T<:Number","page":"Internal","title":"FiniteHorizonGramians._symmetrize!","text":"_symmetrize!(A::AbstractMatrix{T}) where {T<:Number}\n\nDiscards the skew-Hermitian part of A in-place.\n\n\n\n\n\n","category":"method"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = FiniteHorizonGramians","category":"page"},{"location":"#FiniteHorizonGramians","page":"Home","title":"FiniteHorizonGramians","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for FiniteHorizonGramians.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"}]
}
