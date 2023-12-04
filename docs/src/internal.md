# Internal methods

```@docs
FiniteHorizonGramians._exp_and_gram_chol_init!(eA::AbstractMatrix{T}, U::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}, method::ExpAndGram{T,q}, cache = nothing) where {T,q}
FiniteHorizonGramians._exp_and_gram_double!(eA, U, s, cache)
FiniteHorizonGramians._dims_if_compatible(A::AbstractMatrix, B::AbstractMatrix)
FiniteHorizonGramians.triu2cholesky_factor!(A::AbstractMatrix{T}) where {T<:Number}
FiniteHorizonGramians._symmetrize!(A::AbstractMatrix{T}) where {T<:Number}
```