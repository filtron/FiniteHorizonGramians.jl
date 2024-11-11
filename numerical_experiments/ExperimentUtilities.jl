module ExperimentUtilities

using LinearAlgebra,
    Random,
    SpecialFunctions,
    ExponentialUtilities,
    MatrixDepot,
    FiniteHorizonGramians,
    TuePlots

import FiniteHorizonGramians as FHG


struct MatrixFraction <: AbstractExpAndGramAlgorithm end
export MatrixFraction

"""
FiniteHorizonGramians.exp_and_gram(A::AbstractMatrix{T}, B::AbstractMatrix{T}, ::MatrixFraction) where {T}


Computes the matrix exponential of A and the finite horizon Grammian of A and B using a non-adaptive
Pade based scaling and squaring method
"""
function FHG.exp_and_gram(
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
    ::MatrixFraction,
) where {T<:Number}
    LinearAlgebra.require_one_based_indexing(A) # to be on the safe side
    LinearAlgebra.require_one_based_indexing(B) # to be on the safe side

    n, _ = size(B)
    n == LinearAlgebra.checksquare(A) || throw(
        DimensionMismatch(
            "size of A, $(LinearAlgebra.checksquare(A)), incompatible with the size of B, $(size(B)).",
        ),
    )

    pre_array = [A B*B'; zero(A) -A']
    method = ExpMethodGeneric()
    cache = ExponentialUtilities.alloc_mem(pre_array, method)

    post_array = exponential!(pre_array, method, cache)
    Φ = post_array[1:n, 1:n]
    G = post_array[1:n, n+1:2n] * Φ'
    FHG._symmetrize!(G)

    return Φ, G
end


"""
    error_ub(A::AbstractMatrix{T}, B::AbstractMatrix{T})


Computes an error estimate of the computed Gramian from the relative condition number.
"""
function error_ub(A, B)
    T = promote_type(eltype(A), eltype(B))
    R = real(T)
    u = eps(R) / R(2) # unit round off
    err = gramcond(A, B) * u
    return err
end


abstract type AbstractExperiment end



struct IntegratorExperiment <: AbstractExperiment
    maxn::Integer
end



function integrator2AB(T, ndiff)
    A = diagm(-1 => ones(T, ndiff))
    B = vcat(one(T), zeros(T, ndiff, 1))
    return A, B
end

function integrator_exp_and_gram_chol(T, ndiff)
    # transition matrix
    irange = collect(0:ndiff)
    g = @. exp(-logfactorial(irange))
    Φ = zeros(T, ndiff + one(ndiff), ndiff + one(ndiff))
    @simd ivdep for col = 0:ndiff
        @simd ivdep for row = col:ndiff
            @inbounds Φ[row+1, col+1] = g[row-col+1]
        end
    end

    # cholesky factor
    L = zeros(T, ndiff + one(ndiff), ndiff + one(ndiff))
    @simd ivdep for n = 0:ndiff
        something = logfactorial(n)
        @simd ivdep for m = 0:ndiff
            if m <= n
                @inbounds L[n+1, m+1] = exp(
                    something + T(0.5) * log(2 * m + 1) - logfactorial(n - m) -
                    logfactorial(n + m + 1),
                )
            end
        end
    end
    U = permutedims(L)
    return Φ, U
end

function integrator_exp_and_gram(T, ndiff)
    Φ, U = integrator_exp_and_gram_chol(T, ndiff)
    G = U' * U
    FHG._symmetrize!(G)
    return Φ, G
end

function run_experiment(e::IntegratorExperiment)
    T = Float64
    alg = AdaptiveExpAndGram{T}()
    itr = 1:e.maxn

    relerrs_G = zeros(T, length(itr))
    relerrs_U = zeros(T, length(itr))
    relerr_ub = zeros(T, length(itr))
    for (i, ndiff) in pairs(itr)
        Araw, Braw = integrator2AB(T, ndiff)

        _, G_gt = integrator_exp_and_gram(BigFloat, big(ndiff))
        _, G = exp_and_gram(Araw, Braw, alg)


        _, U_gt = integrator_exp_and_gram_chol(BigFloat, big(ndiff))
        _, U = exp_and_gram_chol(Araw, Braw, alg)

        G_gt = T.(G_gt)
        U_gt = T.(U_gt)
        relerrs_G[i] = opnorm(G - G_gt, 2) / opnorm(G_gt, 2)
        relerrs_U[i] = opnorm(U - U_gt, 2) / opnorm(U_gt, 2)
        relerr_ub[i] = error_ub(Araw, Braw)
    end
    return relerrs_G, relerrs_U, relerr_ub
end



const EXCLUDED_NAMES = [
    "blur",
    "hadamard",
    "phillips",
    "rosser",
    "neumann",
    "parallax",
    "poisson",
    "wathen",
    "invhilb",
    "vand",
    "golub",
    "magic",
    "pascal",
]

struct MatrixDepotBuiltIn{T} <: AbstractExperiment
    n::Integer
    m::Integer
    names::T
end

function MatrixDepotBuiltIn(n::Integer, m::Integer)
    matrix_names = mdlist(:builtin)
    matrix_names = [name for name in matrix_names if !in(name, EXCLUDED_NAMES)]
    return MatrixDepotBuiltIn{typeof(matrix_names)}(n, m, matrix_names)
end

export MatrixDepotBuiltIn

function run_experiment(rng::AbstractRNG, e::MatrixDepotBuiltIn)
    T = Float64
    alg_gt = MatrixFraction()
    alg = AdaptiveExpAndGram{T}()

    relerrs = zeros(T, length(e.names))
    relerr_ub = zeros(T, length(e.names))
    for (i, name) in pairs(e.names)
        Araw, Braw = Matrix(matrixdepot(name, e.n)), randn(rng, e.n, e.m)
        Braw = Braw / opnorm(Braw, 2)
        A, B = big.(Araw), big.(Braw)

        _, G_gt = exp_and_gram(A, B, alg_gt)
        _, G = exp_and_gram(Araw, Braw, alg)

        G_gt = T.(G_gt)
        relerrs[i] = opnorm(G - G_gt, 2) / opnorm(G_gt, 2)
        relerr_ub[i] = error_ub(Araw, Braw)
    end
    return relerrs, relerr_ub
end

run_experiment(e::MatrixDepotBuiltIn) = run_experiment(Random.default_rng(), e)


struct LaguerreExperiment{T} <: AbstractExperiment
    λ::T
    maxn::Integer
end

LaguerreExperiment(λ::T) where {T} = LaguerreExperiment{T}(λ)

function laguerre2AB(λ::T, n::Integer) where {T}
    A = λ * (I - T(2) * tril(ones(T, n, n)))
    B = sqrt(T(2) * λ) * ones(T, n, 1)
    return A, B
end

function balancedssm_exp_and_gram(A::AbstractMatrix{T}) where {T}
    method = ExpMethodGeneric()
    cache = ExponentialUtilities.alloc_mem(A, method)
    Φ = exponential!(A, method, cache)
    G = I - Φ * Φ'
    FHG._symmetrize!(G)
    return Φ, G
end

function lyap_residual(A, B, eA, G)
    eAB = eA * B
    res = A * G + G * adjoint(A) + (B * adjoint(B) - eAB * adjoint(eAB))
    return res
end

function run_experiment(e::LaguerreExperiment)
    T = Float64
    alg = AdaptiveExpAndGram{T}()
    relerrs = zeros(T, e.maxn)
    relerr_ub = zeros(T, e.maxn)
    for i = 1:e.maxn
        Araw, Braw = laguerre2AB(e.λ, i)
        eA_gt, G_gt = balancedssm_exp_and_gram(big.(Araw))
        eA, G = exp_and_gram(Araw, Braw, alg)


        G_gt = T.(G_gt)
        relerrs[i] = opnorm(G - G_gt, 2) / opnorm(G_gt, 2)
        relerr_ub[i] = error_ub(Araw, Braw)
    end
    return relerrs, relerr_ub
end




export exp_and_gram
export IntegratorExperiment
export MatrixDepotPathological
export LaguerreExperiment
export run_experiment


figdir = joinpath(@__DIR__, "../../note/figs")
if isdir(figdir)
    const FIG_DIR = joinpath(@__DIR__, "../../note/figs")
else
    const FIG_DIR = @__DIR__
end
export FIG_DIR

const STANDARD_WIDTH_PT = 424.06033

function standard_setting()
    width_pt = STANDARD_WIDTH_PT
    width_in = width_pt / TuePlots.POINTS_PER_INCH

    setting = TuePlots.TuePlotsSetting(
        width = width_in,
        width_half = 0.5 * width_in,
        base_fontsize = 11,
        font = "San Serif",
    )
    return setting
end

export standard_setting

end
