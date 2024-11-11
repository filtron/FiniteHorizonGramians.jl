import Pkg
Base.active_project() != joinpath(@__DIR__, "Project.toml") && Pkg.activate(@__DIR__)
haskey(Pkg.project().dependencies, "FiniteHorizonGramians") ||
    Pkg.develop(path = joinpath(@__DIR__, "../"))
isfile(joinpath(@__DIR__, "Manifest.toml")) && Pkg.resolve()
Pkg.instantiate()

using LinearAlgebra, BenchmarkTools
using FastGaussQuadrature, ExponentialUtilities
using FiniteHorizonGramians
import FiniteHorizonGramians as FHG

function matrix_fraction!(eA, G, A, B, t, method, expcache, work_array)
    n = LinearAlgebra.checksquare(A)
    @views begin
        work_array[1:n, 1:n] .= A
        mul!(work_array[1:n, n+1:2n], B, B')
        work_array[n+1:2n, 1:n] .= zero(eltype(A))
        work_array[n+1:2n, n+1:2n] .= -A'
        work_array .= work_array * t
        W = exponential!(work_array, method, expcache)
        eA .= W[1:n, 1:n]
        G = mul!(G, W[1:n, n+1:2n], eA')
    end
    FHG._symmetrize!(G)
    return eA, G
end

function quadrature_disc!(eA, U, A, B, t, method, expcache, work_array)
    m, n = size(B)

    nodes, weights = gausslegendre(m)
    b, a = t, 0
    @. nodes = (b - a) / 2 * nodes + (a + b) / 2
    @. weights = (b - a) / 2 * weights

    for i in eachindex(nodes)
        dt = nodes[i]
        @. eA = A * dt
        exponential!(eA, method, expcache)
        mul!(view(work_array, (i-1)*n+1:i*n, 1:m), B', eA, sqrt(weights[i]), false)
    end

    # gausslegendre does not include end-point
    # might be better to use Radau/Lobatto or whatever it is called...
    @. eA = A * t
    eA = LinearAlgebra.exp!(eA)
    U .= qr!(work_array).R

    return eA, U
end


function main()
    m = 20
    n = 1
    Ts = (Float32, Float64)
    Q = 13

    for T in Ts
        A = I - 2 * tril(ones(T, m, m))
        B = T(sqrt(2)) * ones(T, m, n)
        t = one(T)

        method1 = ExpAndGram{T,Q}()
        cache1 = FiniteHorizonGramians.alloc_mem(A, B, method1)

        method2 = ExpMethodHigham2005()
        expcache2 = ExponentialUtilities.alloc_mem(similar(A, 2m, 2m), method2)
        work_array2 = similar(A, 2m, 2m)

        method3 = ExpMethodHigham2005()
        expcache3 = ExponentialUtilities.alloc_mem(A, method3)
        work_array3 = similar(A, m * n, m)

        eA, U, G = similar(A), similar(A), similar(A)
        println("\nBenchmark | T = $(T), size(B) = $(size(B))")

        println("\nExpAndGram")
        @btime exp_and_gram_chol!($eA, $U, $A, $B, $t, $method1, $cache1)

        println("MatrixFraction")
        @btime matrix_fraction!($eA, $G, $A, $B, $t, $method2, $expcache2, $work_array2)

        println("Quadrature")
        @btime quadrature_disc!($eA, $U, $A, $B, $t, $method3, $expcache3, $work_array3)

    end
end
main()
