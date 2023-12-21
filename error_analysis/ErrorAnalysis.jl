module ErrorAnalysis

using LinearAlgebra, TaylorSeries, Jacobi, NonlinearSolve, PolynomialRoots, TuePlots

"""
    pade_den(q::Integer, θ)

Computes the Pade denominator.
"""
function pade_den(q::Integer, θ)
    den = sum(0:q) do j
        binomial(q, j) * factorial(2 * q - j) * (-θ)^j
    end
    return den / factorial(2 * q)
end

"""
    pade(q::Integer, θ)

Computes the Pade approximation. 
"""
function pade(q::Integer, θ)
    num = pade_den(q, -θ)
    den = pade_den(q, θ)
    return num / den
end

"""
    pade_analytic_radius(T, q::Integer)

Computes the maximum radius around origin for which the Pade denominator 
is analytic. 
"""
function pade_analytic_radius(T, q::Integer)
    coeff = map(0:q) do j
        binomial(q, j) * factorial(2 * q - j) * (-one(T))^j
    end
    minimum(abs.(roots(coeff)))
end


"""
    pade_den_bound(T, q::Integer, order::Integer, θ)

Computes a norm bound for the inverse of the Pade denominator 
assuming the argument has norm at most θ. 
"""
function pade_den_bound(T, q::Integer, order::Integer, θ)
    dθ = Taylor1(T, order)
    series = exp(dθ / T(2)) * pade_den(q, dθ) - one(T)
    coeffs = abs.([getcoeff(series, i) for i = 0:order])
    bound = Taylor1(coeffs, order)
    return exp(θ / 2) / (one(T) - bound(θ))
end

"""
    BackwardBound{T,A}

Callable struct for computing the backward bound for the matrix exponential. 
"""
struct BackwardBound{T,A}
    f::A
    function BackwardBound{T}(q::Integer, order::Integer) where {T}
        dθ = Taylor1(T, order)
        ρ = exp(-dθ) * pade(q, dθ) - one(T)
        coeffs = abs.([i < 2 * q + 1 ? zero(T) : getcoeff(ρ, i) for i = 0:order])
        f = Taylor1(coeffs, order)
        new{T,typeof(f)}(f)
    end
end

function (BWB::BackwardBound{T,A})(θ::T)::T where {T,A}
    -log(one(T) - BWB.f(θ)) / θ
end


""""
    backward_bound_exp(T, qs, order::Integer)

Compute the least norm that guarantees unit roundoff error in 
the computed matrix exponential. 
"""
function backward_bound_exp(PT, T, qs, order::Integer)
    # twice the result of Higham (2005) is a good upper bound
    if PT <: Float64
        scaling = T(2)
    elseif PT <: Float32
        scaling = T(3.2)
    end
    unit_roundoff = eps(real(PT)) / 2
    ubs =
        T.([
            3.7e-8,
            5.3e-4,
            1.5e-2,
            8.5e-2,
            2.5e-1,
            5.4e-1,
            9.5e-1,
            1.5e0,
            2.1e0,
            2.8e0,
            3.6e0,
            4.5e0,
            5.4e0,
            6.3e0,
            7.3e0,
            8.4e0,
            9.4e0,
            1.1e1,
            1.2e1,
            1.3e1,
            1.4e1,
        ]) * scaling
    alg = Bisection()
    θs = map(zip(qs, ubs)) do (q, ub)
        bwb = BackwardBound{T}(q, order)
        fun(θ, p) = bwb(θ) - T(unit_roundoff)
        interval = [big(1e-10), ub]
        prob = IntervalNonlinearProblem(fun, interval)
        sol = solve(prob, alg)
        sol.u
    end
    return θs
end


"""
    V(q::Integer, t, θ)

Computes V_q(t, θ)
"""
function V(q::Integer, t, θ)
    θt = θ * t
    expθt = exp(θt)
    den = pade_den(q, θ)
    const_factor = (-1)^q * factorial(q) / factorial(2 * q) / den / expθt
    v = sum(0:q) do k
        factor = binomial(q, k) * binomial(q + k, k) * factorial(k) * (-θ)^(q - k)
        term = sum(0:k) do m
            θt^m / factorial(m)
        end
        factor * (term - expθt)
    end
    return v * const_factor
end

"""
    legzeros(q::Integer, T)

Computes the zeros of the shifted Legendre polynomials. 
"""
function legzeros(q::Integer, T)
    zs = zeros(T, q)
    jacobi_zeros!(q, zero(T), zero(T), zs)
    zs .= (zs .+ one(eltype(zs))) / eltype(zs)(2)
    return zs
end

struct VBound{T<:Number,C}
    q::Integer
    order::Integer
    series::C
    function VBound{T}(q::Integer, order::Integer, ngrid::Integer) where {T}
        zs = legzeros(q, T)

        # create dense grid including Legendre zeros 
        # ngrid might have to be at least 2 for this to not break... 
        end_points = zs
        pushfirst!(end_points, zero(T))
        push!(end_points, one(T))
        #intervals = zip(end_points[begin:end-1], end_points[begin+1:end])
        #time_points =
        #    mapreduce(x -> LinRange(first(x), last(x), ngrid), vcat, intervals) |> unique
        time_points = zs
        dθ = Taylor1(T, order)
        series = map(time_points) do tp
            te = V(q, tp, dθ)
            coeffs = abs.([i < q + 1 ? zero(T) : getcoeff(te, i) for i = 0:order])
            Taylor1(coeffs, order)
        end
        new{T,typeof(series)}(q, order, series)
    end
end

function (VB::VBound{T,C})(θ::T) where {T,C}
    evals = map(VB.series) do f
        f(θ)
    end
    return maximum(evals)
end


"""
    backward_bound_gram(order::Integer)

Computes bounds ηs such that the backward error in G is less than unit roundoff in Float64. 
"""
function backward_bound_gram(PT, T, qs, order::Integer, ngrid::Integer)
    # fraction of θs probably good upper bound for ηs 
    unit_roundoff = eps(real(PT)) / 2
    ubs = backward_bound_exp(PT, T, qs, order) / T(1.0)
    ηs = similar(ubs)
    T = eltype(ηs)
    alg = Bisection()
    ηs = map(zip(qs, ubs)) do (q, ub)
        vb = VBound{T}(q, order, ngrid)
        fun(θ, p) = vb(θ) - T(unit_roundoff)
        interval = [big(0.0), ub]
        prob = IntervalNonlinearProblem(fun, interval)
        sol = solve(prob, alg)
        sol.u
    end
    return ηs
end

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
