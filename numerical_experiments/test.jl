# activates the script environment and instantiates it 
import Pkg
Base.active_project() != joinpath(@__DIR__, "Project.toml") && Pkg.activate(@__DIR__)
haskey(Pkg.project().dependencies, "FiniteHorizonGramians") ||
    Pkg.develop(path = joinpath(@__DIR__, "../"))
isfile(joinpath(@__DIR__, "Manifest.toml")) && Pkg.resolve()
Pkg.instantiate()

includet("ExperimentUtilities.jl")
using FiniteHorizonGramians, .ExperimentUtilities
using BenchmarkTools

λ = 1e-0
n = 100
A, B = ExperimentUtilities.laguerre2AB(λ, n)
T = Float64

q = 13
method = ExpAndGram{T,q}()

#@btime SSXtras._exp_and_gram_chol_init($A, $B, $le)
#@btime SSXtras._exp_and_gram_chol_init($A, $B, $method)
#@btime exp_and_gram_chol(A, B, method)
Φ, U = exp_and_gram_chol(A, B, method)

#@btime Φ, U = exp_and_gram_chol(A, B, method)



function main()
n = 10
a0 = 1.0;
as = 1e-6*ones(n-1)
A = diagm(-1 => -as) + diagm(1 => as) 
A[1, 1] = -a0 
B = [sqrt(2a0); zeros(n-1);;]
C = [one(a0) zeros(1, n-1)]


ωs = 2π * LinRange(0, 1.0,2^8+1)
psd = zeros(complex(eltype(ωs)), length(ωs)) 

for (i, ω) in pairs(ωs)
    psd[i] = C *( (im*ω*I - A) \ B ) |> first 
end
return ωs, psd 
end

#ωs, psd = main()
#lines(ωs, abs2.(psd))

