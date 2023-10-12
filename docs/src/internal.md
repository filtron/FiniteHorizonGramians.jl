# Internal methods 

```@docs
FiniteHorizonGramians._exp_and_gram_chol_init(A::AbstractMatrix{T}, B::AbstractMatrix{T}, method::ExpAndGram{T,q}) where {T,q}
FiniteHorizonGramians._dims_if_compatible(A::AbstractMatrix, B::AbstractMatrix)
FiniteHorizonGramians.triu2cholesky_factor!(A::AbstractMatrix{T}) where {T<:Number}
FiniteHorizonGramians._symmetrize!(A::AbstractMatrix{T}) where {T<:Number}
```