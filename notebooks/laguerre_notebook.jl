### A Pluto.jl notebook ###
# v0.19.29

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 81050e86-fed3-11ed-025f-8de03833c11f
begin
    import Pkg
    Base.active_project() != joinpath(@__DIR__, "Project.toml") && Pkg.activate(@__DIR__) # activate the project in directory of file 
    haskey(Pkg.project().dependencies, "SSXtras") ||
        Pkg.develop(path = joinpath(@__DIR__, "../")) # add parent project as a dependency 
    isfile(joinpath(@__DIR__, "Manifest.toml")) && Pkg.resolve() # probably not needed 
    Pkg.instantiate()

    using SSXtras, LinearAlgebra, CairoMakie, Markdown, InteractiveUtils, Pluto, PlutoUI
end

# ╔═╡ 19887583-96ca-45c5-9298-59ca82ec3eb6
html"""
<style>
	main {
		margin: 0 auto;
		max-width: 2000px;
    	padding-left: max(160px, 15%);
    	padding-right: max(160px, 15%);
	}
</style>
"""

# ╔═╡ 01886496-87e8-4a6a-be63-f2bed179540a
md"""
# Computing Cholesky factors of finite horizon Grammians

## Problem statement

Given a pair of matrices $A \in \mathbb{R}^n$ and $B \in \mathbb{R}^{n \times m}$, there are two matrix functions $\Phi(A)$ and $G(A, B)$,  

```math 
\begin{aligned}
\Phi(A) &= e^{A}, \\ 
G(A, B) &= \int_0^1 e^{A t} B B^* e^{A^* t} \mathrm{d} t,
\end{aligned}
```
where $\Phi$ is just the standard matrix exponential and $G(A, B)$ is the controllability Grammian of $(A, B)$ on the interval $[0, 1]$.
It is of interest to compute a Choleskii factor, $U(A, B)$, of $G(A, B)$ without forming $G(A, B)$ as an intermediate step. $U(A, B)$ is required in numerically robust implementations of linear estimators in state-space models as well as numerically robust implementations of state-space balancing and truncation. However, computing the Choleskii factorization of $G(A, B)$ directly may fail due to round-off errors. 

In this notebook, a new method is demonstrated that computes $U(A, B)$ directly. 
"""

# ╔═╡ 014c8f36-53c0-400f-ab88-5c16a759f478
md"""

## Setup 

"""

# ╔═╡ 57077420-e68a-4163-87ad-a6a6db2562b7
md""" 

## Implementing a Laguerre network 

The system matrices for a Laguerre network is given by 

```math
\begin{aligned} 
A = -\lambda
\begin{pmatrix} 
1 & 0 & \ldots & \ldots & 0 \\ 
2 & \ddots & \ddots & \ldots & \vdots \\ 
\vdots & \ddots & \ddots & \ddots & \vdots \\
2 & \ldots & \ldots & \ldots  & 1 
\end{pmatrix}, 
\quad 
B = \sqrt{2\lambda}
\begin{pmatrix} 
1 \\ \vdots \\ \vdots \\ 1  
\end{pmatrix}
\end{aligned}
```

Since the system is unput balanced, $A + A^* = - B B^*$, the Grammian may be computed as 

```math 
G(A, B) = I - e^{A} e^{A^*}, 
```
which will serve as a reference solution. 

"""

# ╔═╡ 07dd3620-7811-4524-bdd1-38672dc33df7
function laguerre2AB(λ::T, n::Integer) where {T}
    A = λ * (I - T(2) * tril(ones(T, n, n)))
    B = sqrt(T(2) * λ) * ones(T, n, 1)
    return A, B
end

# ╔═╡ 88cccd6a-2d1c-424b-84cf-04d12970f200
function laguerre_exp_and_gram(λ::T, n::Integer) where {T}
    A, _ = laguerre2AB(λ, n)
    Φ = exp(A)
    G = I - Φ * Φ'
    return Φ, G
end

# ╔═╡ 37f44c34-2c2a-455b-9f05-55537d4250fc
md""" 

## Experiment 

The slider $n$ determines the dimension of the problem, and the slider $\lambda$ determines the eigenvalues of $A$. 

"""

# ╔═╡ 57103d75-4672-4035-bcdb-db9b3b8dd24d

md""" 

`n = ` $(@bind n PlutoUI.Slider(10:5:500, default=10, show_value=true))

`λ = ` $(@bind λ PlutoUI.Slider(1e-3:0.1:1e+2, default=1.0, show_value=true))
	
	"""

# ╔═╡ 682029e1-49be-4eab-834b-4e3130b0217d
begin
    Ggt, U = let
        T = Float64
        A, B = laguerre2AB(λ, n)
        Φgt, Ggt = laguerre_exp_and_gram(λ, n)
        Φ, U = exp_and_gram_chol(A, B, AdaptiveExpAndGram{T}())
        Ggt, U
    end
    nothing
end

# ╔═╡ fdc315ac-2162-43e1-9fe0-6f5ea9503c3e
fig = with_theme(Lines = (; cycle = :linestyle)) do
    fig = Figure(resolution = (800, 600))
    ax1 = Axis(
        fig[1, 1],
        title = "n = $(n), λ = $(round(λ, digits=4))",
        yscale = log10,
        ylabel = "√λₙ",
        xlabel = "n",
    )
    tol = eps(eltype(Ggt)) / 2
    eigvals_ref = eigvals(Ggt)
    eigvals_ref =
        [eigvals_ref[i] < 0 ? tol : eigvals_ref[i] for i in eachindex(eigvals_ref)]
    eigvals_ref = sqrt.(eigvals_ref)

    eigvals_new = sort(svdvals(U))
    lines!(ax1, 1:n, eigvals_ref, color = "black", label = "reference")
    lines!(ax1, 1:n, eigvals_new, color = "black", label = "new method")
    lines!(ax1, 1:n, fill(sqrt(tol), n), color = "black", label = "√ε/2")
    axislegend()
    fig
end

# ╔═╡ 91f3891b-06b2-41bb-ac2e-6c279a05587a
logabsdet(Ggt)[1], 2logdet(U)

# ╔═╡ Cell order:
# ╟─19887583-96ca-45c5-9298-59ca82ec3eb6
# ╟─01886496-87e8-4a6a-be63-f2bed179540a
# ╟─014c8f36-53c0-400f-ab88-5c16a759f478
# ╠═81050e86-fed3-11ed-025f-8de03833c11f
# ╟─57077420-e68a-4163-87ad-a6a6db2562b7
# ╠═07dd3620-7811-4524-bdd1-38672dc33df7
# ╠═88cccd6a-2d1c-424b-84cf-04d12970f200
# ╠═37f44c34-2c2a-455b-9f05-55537d4250fc
# ╟─57103d75-4672-4035-bcdb-db9b3b8dd24d
# ╠═682029e1-49be-4eab-834b-4e3130b0217d
# ╠═fdc315ac-2162-43e1-9fe0-6f5ea9503c3e
# ╠═91f3891b-06b2-41bb-ac2e-6c279a05587a
