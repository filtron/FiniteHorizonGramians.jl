# FiniteHorizonGramians

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://filtron.github.io/FiniteHorizonGramians.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://filtron.github.io/FiniteHorizonGramians.jl/dev/)
[![Build Status](https://github.com/filtron/FiniteHorizonGramians.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/filtron/FiniteHorizonGramians.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/filtron/FiniteHorizonGramians.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/filtron/FiniteHorizonGramians.jl)




## Installation 

```julia 
] add https://github.com/filtron/FiniteHorizonGramians.jl.git
```

## Usage 

```julia 
A = [0.0 0.0; 1.0 0.0]
B = [1.0; 0.0;;]

expA, U = exp_and_gram_chol(A, B)
_, G = exp_and_gram(A, B)

G ≈ U' * U # returns true 
```
