# FiniteHorizonGramians

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://filtron.github.io/FiniteHorizonGramians.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://filtron.github.io/FiniteHorizonGramians.jl/dev/)
[![Build Status](https://github.com/filtron/FiniteHorizonGramians.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/filtron/FiniteHorizonGramians.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/filtron/FiniteHorizonGramians.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/filtron/FiniteHorizonGramians.jl)

## Introduction

Associated with the pair of matrices $A \in \mathbb{R}^{n \times n}$ and $B \in \mathbb{R}^{n \times m}$ is the differential Lyapunov equation:

$$
\dot{Q}(\tau) = A Q(\tau) + Q(\tau) A^* + B B^*, \quad Q(0) = 0,\quad \tau \in [0, t].
$$

$Q(t)$ is the finite Horizon controllability Gramian of $(A, B)$, over the interval $[0, t]$,
and may be equivalently expressed by the function $Q(t) = G(At, B \sqrt{t})$, where

$$
G(A, B) = \int_0^1 e^{A t} B B^* e^{A^* t} \mathrm{d} t.
$$

The Gramian, $G(A, B)$, is positive semi-definite and therefore has an upper triangular Cholesky factor, $U(A, B)$.
This package provides algorithms for computing both $e^A$ and $U(A, B)$,
without having to form $G(A, B)$ as an intermediate step.
This avoids the problem of failing Cholesky factorizations when computing $G(A, B)$ directly leads to a numerically non-positive definite matrix.

## Application

Consider the Gauss-Markov process

$$
\dot{x}(t) = A x(t) + B \dot{w}(t).
$$

It can be shown that the process $x$ has a transition density given by

$$
p(t + h, x \mid t, x') = \mathcal{N} \(x; e^{A h} x', G(A h, \sqrt{h} B) \).
$$

This package offers a method to both compute the transition matrix $e^{A h}$ and a Cholesky factor of $G(A h, \sqrt{h} B)$ in a numerically robust way.
This is useful for instance in so called array implementations of Gauss-Markov regression (i.e. square-root Kalman filters etc).

## Installation

```julia
] add FiniteHorizonGramians
```

## Basic usage

```julia
using FiniteHorizonGramians, LinearAlgebra

n = 15
A = (I - 2.0 * tril(ones(n, n)))
B = sqrt(2.0) * ones(n, 1)
ts = LinRange(0.0, 1.0, 2^3)

method = ExpAndGram{Float64,13}()
cache = FiniteHorizonGramians.alloc_mem(A, B, method) # allocate memory for intermediate calculations
eA, U = similar(A), similar(A) # allocate memory for outputs

for t in ts
   exp_and_gram_chol!(eA, U, A, B, t, method, cache)
    # do cool stuff with eA and U here?
end

eA, G = exp_and_gram!(eA, similar(U), A, B, last(ts), method, cache) # we can comput the full Gramian if we prefer
G ≈ U'U # true
eA ≈ exp(A * last(ts)) # we also get an accurate matrix exponential, of course.
```

## Related packages
* [MatrixEquations.jl](https://github.com/andreasvarga/MatrixEquations.jl) provides an algorithm for computing the Cholesky factor of infinite horizon Gramians (i.e. solutions to algebraic Lyapunov equations).
* [ExponentialUtilities.jl](https://github.com/SciML/ExponentialUtilities.jl) provides various algorithms for matrix exponentials and related quantities.
