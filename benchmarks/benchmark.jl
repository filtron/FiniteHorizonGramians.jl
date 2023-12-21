import Pkg
Base.active_project() != joinpath(@__DIR__, "Project.toml") && Pkg.activate(@__DIR__)
haskey(Pkg.project().dependencies, "FiniteHorizonGramians") ||
    Pkg.develop(path = joinpath(@__DIR__, "../"))
isfile(joinpath(@__DIR__, "Manifest.toml")) && Pkg.resolve()
Pkg.instantiate()

using LinearAlgebra, BenchmarkTools
using FastGaussQuadrature
using FiniteHorizonGramians
import FiniteHorizonGramians as FHG

function matrix_fraction!(eA, G, A, B, t, cache = similar(A, sum(size(A)), sum(size(A))))
    n = LinearAlgebra.checksquare(A)
    @views begin
        cache[1:n, 1:n] .= A
        mul!(cache[1:n, n+1:2n], B, B')
        cache[n+1:2n, 1:n] .= zero(eltype(A))
        cache[n+1:2n, n+1:2n] .= -A'
        cache .= cache * t
        W = LinearAlgebra.exp!(cache)
        eA .= W[1:n, 1:n]
        G = mul!(G, W[1:n, n+1:2n], eA')
    end
    FHG._symmetrize!(G)
    return eA, G
end

function quadrature_disc!(eA, U, A, B, t, nodes, weights, R)
    m, n = size(B)

    nodes, weights = gausslegendre(m)
    b, a = t, 0
    @. nodes = (b - a) / 2 * nodes + (a + b) / 2
    @. weights = (b - a) / 2 * weights


    for i in eachindex(nodes)
        dt = nodes[i]
        @. eA = A * dt
        eA = LinearAlgebra.exp!(eA)
        mul!(view(R, (i-1)*n+1:i*n, 1:m), B', eA, sqrt(weights[i]), false)
    end

    # gausslegendre does not include end-point
    # might be better to use Radau/Lobatto or whatever it is called...
    @. eA = A * t
    eA = LinearAlgebra.exp!(eA)
    U .= qr!(R).R

    return eA, U
end


function main()
    m = 20
    n = 1
    Ts = (Float32, Float64)
    Q = 9

    for T in Ts
        A = I - 2 * tril(ones(T, m, m))
        B = T(sqrt(2)) * ones(T, m, n)
        t = one(T)
        method = ExpAndGram{T,Q}()

        mcache = FiniteHorizonGramians.alloc_mem(A, B, method)

        mfcache = similar(A, 2m, 2m)


        qcache = similar(A, m*n, m)

        eA, U, G = similar(A), similar(A), similar(A)
        println("\nBenchmark | T = $(T), m = $(m), n = $(n)")

        println("\nExpAndGram")
        @btime exp_and_gram_chol!($eA, $U, $A, $B, $t, $method, $mcache)

        println("\nQuadrature")
        @btime quadrature_disc!($eA, $U, $A, $B, $t, $nodes, $weights, $qcache)

        println("\nMatrixFraction")
        @btime matrix_fraction!($eA, $G, $A, $B, $t, $mfcache)
    end
end
main()