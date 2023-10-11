# FiniteHorizonGramians

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://filtron.github.io/FiniteHorizonGramians.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://filtron.github.io/FiniteHorizonGramians.jl/dev/)
[![Build Status](https://github.com/filtron/FiniteHorizonGramians.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/filtron/FiniteHorizonGramians.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/filtron/FiniteHorizonGramians.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/filtron/FiniteHorizonGramians.jl)

Associated with the pair of matrices $A \in \mathbb{R}^{n \times n}$ and $B \in \mathbb{R}^{n \times m}$ is the finite horizon controllability Gramian over the unit interval 

$$
G(A, B) = \int_0^1 e^{A t} B B^* e^{A^* t} \mathrm{d} t. 
$$

The Gramian, $G(A, B)$, is positive semi-definite and therefore has an upper triangular Cholesky factor, $U(A, B)$,
such that $G(A, B) = U^*(A, B) U(A, B)$. 
This package provides algorithms for computing both the matrix exponential, $e^A$, and the Cholesky factor, $U(A, B)$,
without having to form $G(A, B)$ as an intermediate step. 
This avoids the problem of failing Cholesky factorization when computing $G(A, B)$ directly leads to a numerically non-positive definite matrix. 

## Application 

Consider the Gauss-Markov process 

$$
\dot{x}(t) = A x(t) + B \dot{w}(t).
$$

It can be shown that the process $x$ has a transition density given by 

$$ 
p(t + h, x \mid t, x') = \mathcal{N}\big(x; e^{A h} x', G(A h, \sqrt{h} B) \big). 
$$

Consequently this package offers a method to both compute the transition matrix $e^{A h}$ and a Cholesky factor of $G(A h, \sqrt{h} B)$ in a numerically robust way.
This is useful for instance in so called array implementations of Gauss-Markov regression (i.e. square-root Kalman filters etc).

## Installation 

```julia 
] add https://github.com/filtron/FiniteHorizonGramians.jl.git
```

## Basic usage 

```julia 
A = [0.0 0.0; 1.0 0.0]
B = [1.0; 0.0;;]

method = AdaptiveExpAndGram{Float64}()
expA, U = exp_and_gram_chol(A, B, method)
_, G = exp_and_gram(A, B, method)

G ≈ U' * U # returns true 
```

## Related packages 
* [MatrixEquations.jl](https://github.com/andreasvarga/MatrixEquations.jl) provides an algorithm for computing the Cholesky factor of infinite horizon Gramians (i.e. solutions to algebraic Lyapunov equations).
* [ExponentialUtilities.jl](https://github.com/SciML/ExponentialUtilities.jl) provides various algorithms for matrix exponentials and related quantities.
