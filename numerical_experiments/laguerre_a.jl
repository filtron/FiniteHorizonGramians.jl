# activates the script environment and instantiates it
import Pkg
Base.active_project() != joinpath(@__DIR__, "Project.toml") && Pkg.activate(@__DIR__)
haskey(Pkg.project().dependencies, "FiniteHorizonGramians") ||
    Pkg.develop(path = joinpath(@__DIR__, "../"))
isfile(joinpath(@__DIR__, "Manifest.toml")) && Pkg.resolve()
Pkg.instantiate()

using Revise
includet("ExperimentUtilities.jl")
using FiniteHorizonGramians, Random, .ExperimentUtilities
using MakieCore, CairoMakie, TuePlots, LaTeXStrings


using LinearAlgebra, FiniteHorizonGramians

function mf_gram(A, B)
    n, _ = size(B)
    pre_array = [A B*B'; zero(A) -A']
    post_array = exp(pre_array)
    Φ = post_array[1:n, 1:n]
    G = post_array[1:n, (n+1):2n] * Φ'
    G = hermitianpart!(G)
    return G
end

function lyap_gram(A, B)
    eA = exp(A)
    eAB = eA * B
    Q = B * adjoint(B) - eAB * adjoint(eAB)
    G = lyap(A, Q)
    return G
end

function eigen_gramchol1(A, B)
    eA = exp(A)
    eAB = eA * B
    Q = B * adjoint(B) - eAB * adjoint(eAB)
    G = lyap(A, Q)

    E = eigen(Hermitian(G))
    clipped_eigs = max.(E.values, zero(eltype(E.values)))
    U = qr(Diagonal(sqrt.(clipped_eigs)) * adjoint(E.vectors)).R
    return U
end

function eigen_gramchol2(A, B)
    G = mf_gram(A, B)

    E = eigen(Hermitian(G))
    clipped_eigs = max.(E.values, zero(eltype(E.values)))
    U = qr(Diagonal(sqrt.(clipped_eigs)) * adjoint(E.vectors)).R
    return U
end



function ss_gramchol(A, B)
    T = promote_type(eltype(A), eltype(B))
    alg = AdaptiveExpAndGram{T}()
    _, U = exp_and_gram_chol(A, B, alg)
    return U
end

function ref_gram(A, B)
    eA = exp(A)
    G = I - eA * adjoint(eA)
end


function error(G, U)
    any(isnan, U) && return NaN
    return opnorm(adjoint(U) * U - G, 2) / opnorm(G, 2)
end

function rank_estimate(U, tol)
    any(isnan, U) && return 0
    return rank(U, atol = tol)
end


tol = eps(Float64) / 2 # all singular values smaller than unit round-off are considered zero

function error_estimate(A, B)
    T = promote_type(eltype(A), eltype(B))
    R = real(T)
    u = eps(R) / R(2) # unit round off
    err = 2 * (u * opnorm(A, 2) - log(1 - u))
    return err
end


function main(ns, λs, methods)
    relerrs = zeros(Float64, length(λs), length(ns), length(methods))
    ranks = zeros(Int64, length(λs), length(ns), length(methods))
    err_estimates = zeros(Float64, length(λs), length(ns))

    for (i, λ) in pairs(λs)
        for (j, n) in pairs(ns)
            A = λ * (I - 2 * tril(ones(n, n)))
            B = sqrt(2 * λ) * ones(n, 1)
            G = ref_gram(A, B)
            err_estimates[i, j] = error_estimate(A, B)
            for (k, method) in pairs(methods)
                U = method(A, B)
                e = error(G, U)
                r = rank_estimate(U, tol)
                relerrs[i, j, k] = e
                ranks[i, j, k] = r
            end
        end
    end
    return relerrs, ranks, err_estimates
end


nmax = 200
ns = 1:nmax


use_eig1 = true

if use_eig1
    methods = (eigen_gramchol1, ss_gramchol)
    labels = (LaTeXString("EIG1"), LaTeXString("SS"))
    err_fig_name = "laguerre_err1.pdf"
    rank_fig_name = "laguerre_rank1.pdf"
else
    methods = (eigen_gramchol2, ss_gramchol)
    labels = (LaTeXString("EIG2"), LaTeXString("SS"))
    err_fig_name = "laguerre_err2.pdf"
    rank_fig_name = "laguerre_rank2.pdf"
end
linestyles = (:dot, :solid)

λs = (0.5, 1.5, 5.0)

relerrs, ranks, err_estimates = main(ns, λs, methods)

setting = standard_setting()
T = MakieCore.Theme(setting, ncols = length(λs), nrows = 1)

fig_err = with_theme(T) do
    fig = Figure()
    ga = fig[1, 1] = GridLayout()
    axs = Vector{Axis}(undef, length(λs))
    for i in keys(λs)
        ylb = i == 1 ? LaTeXString("error") : ""
        ax = Axis(
            ga[1, i],
            yscale = log10,
            title = LaTeXString("\$ λ = $(λs[i]) \$"),
            ylabel = ylb,
        )
        axs[i] = ax
        lines!(
            ax,
            ns,
            err_estimates[i, :],
            color = "black",
            label = LaTeXString("ERREST"),
            linestyle = :dash,
        )
        i == 1 || hideydecorations!(ax; grid = false)
        for j in keys(methods)
            lines!(
                ax,
                ns,
                relerrs[i, :, j],
                label = labels[j],
                linestyle = linestyles[j],
                color = "black",
            )
        end
    end
    linkyaxes!(axs...)
    colgap!(ga, 10)
    leg = Legend(fig[1, 2], axs[1], framevisible = false)
    #leg.nbanks = 3
    #leg.tellwidth = false
    colsize!(fig.layout, 2, Relative(1 / 5))
    fig
end

figname = joinpath(FIG_DIR, err_fig_name)
Makie.save(figname, fig_err, pt_per_unit = 1.0)
display(fig_err)

fig_rank = with_theme(T) do
    fig = Figure()
    ga = fig[1, 1] = GridLayout()
    axs = Vector{Axis}(undef, length(λs))
    for i in keys(λs)
        ylb = i == 1 ? LaTeXString("rank loss") : ""
        ax = Axis(ga[1, i], title = LaTeXString("\$ λ = $(λs[i]) \$"), ylabel = ylb)
        axs[i] = ax
        #lines!(ax, ns, ns, color = "black", label = LaTeXString("max rank"))
        i == 1 || hideydecorations!(ax; grid = false)
        for j in keys(methods)
            #j != 1 && lines!(ax, ns, ranks[i, :, j], label = labels[j], linestyle = linestyles[j], color = "black")
            lines!(
                ax,
                ns,
                ns - ranks[i, :, j],
                label = labels[j],
                linestyle = linestyles[j],
                color = "black",
            )
        end
    end
    linkyaxes!(axs...)
    colgap!(ga, 10)
    leg = Legend(fig[1, 2], axs[1], framevisible = false)
    #leg.nbanks = 3
    #leg.tellwidth = false
    colsize!(fig.layout, 2, Relative(1 / 5))
    fig
end

figname = joinpath(FIG_DIR, rank_fig_name)
Makie.save(figname, fig_rank, pt_per_unit = 1.0)
display(fig_rank)
