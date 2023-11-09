## Algorithm types

```@docs
AdaptiveExpAndGram{T,A}
ExpAndGram{T,N,A,B,C}
```

## Methods

```@docs
exp_and_gram
exp_and_gram_chol
exp_and_gram_chol!(eA::AbstractMatrix{T}, U::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}, method::AbstractExpAndGramAlgorithm, cache = nothing) where {T<:Number}
exp_and_gram_chol!(eA::AbstractMatrix{T}, _U::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}, t::Number, method::ExpAndGram{T,q}, cache = nothing) where {T<:Number,q}
```
