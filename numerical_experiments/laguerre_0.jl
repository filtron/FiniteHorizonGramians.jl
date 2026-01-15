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


function main(ns, λs)
    relerrs = zeros(Float64, length(λs), length(ns))
    ranks = zeros(Int64, length(λs), length(ns))
    err_estimates = zeros(Float64, length(λs), length(ns))

    for (i, λ) in pairs(λs)
        for (j, n) in pairs(ns)
            A = λ * (I - 2 * tril(ones(n, n)))
            B = sqrt(2 * λ) * ones(n, 1)
            G = ref_gram(A, B)
            err_estimates[i, j] = error_estimate(A, B)
            U = ss_gramchol(A, B)
            e = error(G, U)
            r = rank_estimate(U, tol)
            relerrs[i, j] = e
            ranks[i, j] = r
        end
    end
    return relerrs, ranks, err_estimates
end


nmax = 200
ns = 1:nmax


linestyles = (:dot, :solid)

λs = (0.5, 1.5, 5.0)

relerrs, ranks, err_estimates = main(ns, λs)

gr = (5.0^0.5 - 1.0) / 2.0
ppi = 72.27

#ncols = length(λs)
ncols = 3
nrows = 1
w = 5.867722844887229
h = w * gr / 3
figsize = (ppi * w, ppi * h)

font = "San Serif"
fsz = 11
mainfsz, smallfsz = fsz - 1, fsz - 2
T = Makie.Theme(
    font = font,
    fontsize = fsz,
    size = figsize,
    figure_padding = 5,
    Axis = (
        titlefont = font,
        subtitlefont = font,
        xlabelfont = font,
        ylabelfont = font,
        xticklabelfont = font,
        yticklabelfont = font,
        titlesize = mainfsz,
        subtitlesize = mainfsz,
        xticklabelsize = smallfsz,
        yticklabelsize = smallfsz,
        xlabelsize = mainfsz,
        ylabelsize = mainfsz,
        spinewidth = 0.5,
        xgridwidth = 0.5,
        ygridwidth = 0.5,
        xtickwidth = 0.5,
        ytickwidth = 0.5,
        xticksize = 2,
        yticksize = 2,
    ),
    Legend = (
        titlefont = font,
        labelfont = font,
        labelsize = mainfsz,
        framewidth = 0.5,
        patchsize = (20, 8),
        padding = (5, 5, 5, 5),
    ),
    Label = (font = font, fontsize = smallfsz),
)



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
        Makie.lines!(
            ax,
            ns,
            err_estimates[i, :],
            color = "black",
            linestyle = :dash,
        )
        i == 1 || hideydecorations!(ax; grid = false)
        for j in keys(methods)
            Makie.lines!(
                ax,
                ns,
                relerrs[i, :],
                linestyle = :solid,
                color = "black",
            )
        end
        #ax.xticks = [1, 100, 200]
    end
    linkyaxes!(axs...)
    colgap!(ga, 10)
    #leg.nbanks = 3
    #leg.tellwidth = false
    fig
end

err_fig_name = "laguerre.pdf"
figname = joinpath(FIG_DIR, err_fig_name)
Makie.save(figname, fig_err, pt_per_unit = 1.0)
display(fig_err)

