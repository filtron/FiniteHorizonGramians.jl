using LinearAlgebra, SpecialFunctions
using Test, FiniteHorizonGramians, ExponentialUtilities
using ForwardDiff, FiniteDiff
import FiniteHorizonGramians as FHG
using Aqua, JET

include("test_utils.jl")
include("test_initial_approximations.jl")
include("test_exp_and_gram.jl")
include("test_adaptive_exp_and_gram.jl")
include("test_autodiff.jl")

numeric_types = (Float64,)

@testset "FiniteHorizonGramians.jl" begin

    @testset "initial approximations" begin
        for T in numeric_types
            test_initial_approximation(T)
        end
    end

    @testset "test non-adaptive algorithms" begin
        for T in numeric_types
            test_exp_and_gram(T)
        end
    end


    @testset "adaptive algorithm" begin
        for T in numeric_types
            test_adaptive_exp_and_gram(T)
        end
    end

    @testset "autodiff" begin
        test_ForwardDiff()
    end

    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(FiniteHorizonGramians)
    end

    @testset "Code linting (JET.jl)" begin
        JET.test_package(
            FiniteHorizonGramians;
            target_defined_modules=true,
        )
    end
end
