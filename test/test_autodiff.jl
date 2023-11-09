function test_ForwardDiff()
    reltol = 1e-6 # finite difference Hessian is quite inaccurate
    idxs = 1:1
    qs = [5, 7, 9, 13] # skip q = 3 because very inaccurate 
    @testset "ForwardDiff.jl" begin
        for idx in idxs
            ft, p = diff_problem(Val(idx))
            for q in qs
                f(p) = ft(p, ExpAndGram{eltype(p),q}())
                @test ForwardDiff.gradient(f, p) ≈
                      FiniteDiff.finite_difference_gradient(f, p)
                @test ForwardDiff.hessian(f, p) ≈ FiniteDiff.finite_difference_hessian(f, p) rtol =
                    reltol
            end
            fa(p) = ft(p, AdaptiveExpAndGram{eltype(p)}())
            @test ForwardDiff.gradient(fa, p) ≈ FiniteDiff.finite_difference_gradient(fa, p)
            @test ForwardDiff.hessian(fa, p) ≈ FiniteDiff.finite_difference_hessian(fa, p) rtol =
                reltol
        end
    end
end

function diff_problem(::Val{1})
    p = [-1.5, 2.0]
    function f(p, method)
        A = p[1] * (I - 2 * tril(ones(2, 2)))
        B = p[2] * sqrt(-2 * p[1]) * ones(2, 1)
        v1 = ones(2)
        v2 = [2.0, 0.5]
        Φ, U = exp_and_gram_chol(A, B, method)
        return -logdet(U) - norm(U' \ (v2 - Φ * v1))^2 / 2
    end
    return f, p
end
