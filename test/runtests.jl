using LinearAlgebra, SpecialFunctions
using Test, FiniteHorizonGramians
using ForwardDiff, FiniteDiff
import FiniteHorizonGramians as FHG

include("test_utils.jl")
include("test_initial_approximations.jl")
include("test_adaptive_exp_and_gram.jl")
include("test_autodiff.jl")

@testset "FiniteHorizonGramians.jl" begin

    @testset "initial approximations" begin
        for T in (Float64,)
            test_initial_approximation(T)
        end
    end

    @testset "adaptive algorithm" begin
        for T in (Float64,)
            test_adaptive_exp_and_gram(T)
        end
    end

    @testset "autodiff" begin
        test_ForwardDiff()
    end

end
