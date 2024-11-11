## Algorithm types

```@docs
AdaptiveExpAndGram{T,A}
ExpAndGram{T,N,A,B,C}
```

## Create cache

```@docs
FiniteHorizonGramians.alloc_mem(A, B::AbstractMatrix, ::ExpAndGram{T,q}) where {T,q}
```


## Methods

```@docs
exp_and_gram
exp_and_gram_chol
exp_and_gram!
exp_and_gram_chol!
gramcond
```
