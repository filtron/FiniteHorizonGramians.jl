function test_exp_and_gram(T)
    RT = real(T)

    # this is not totally correct
    if RT <: Float32
        # fp32 tols
        normtols = RT[0.048, 0.61190283, 1.5580196, 2.801222, 5.7639575]
    else
        # fp64 tols
        normtols = RT[
            0.0006794818550677766,
            0.021803366063031033,
            0.13306928209763041,
            0.41098379928173995,
            1.579470165477942,
        ]
    end

    n, m = 8, 2
    L = randn(T, n, n)
    Z = randn(T, n, n)
    A = -L * L' / 2 + (Z - Z') / 2
    B = randn(T, n, m)

    ts = [0.5, 1.0, 1.5]

    qs = [3, 5, 7, 9, 13]
    if RT <: Float32
        # big tol for q = 3 needed for ComplexF32
        tols = [3.0f-3, 1.0f-6, 1.0f-6, 61.0f-7, 6.0f-7]
    else
        tols = [1e-10, 1e-13, 1e-13, 5e-14, 5e-15] # method 3 is quite inaccurate due to the doubling shenanigans...
    end

    zT = zero(RT)

    for (i, q) in enumerate(qs)

        method = ExpAndGram{T,q}()

        err(G1, G2) = opnorm(G1 - G2, 1) / opnorm(G2, 1)

        @testset "q = $(q) | $(T) | exp_and_gram" begin
            @test method.normtol == normtols[i]
            for t in ts
                Φgt, Ggt = mf_exp_and_gram(A, B, t)
                Φ, G = exp_and_gram(A, B, t, method)
                @test isapprox(err(Φ, Φgt), zT, atol = tols[i])
                @test isapprox(err(G, Ggt), zT, atol = tols[i])
                @test ishermitian(G)
            end
            Φgt, Ggt = mf_exp_and_gram(A, B)
            Φ, G = exp_and_gram(A, B, method)
            @test isapprox(err(G, Ggt), zT, atol = tols[i])
        end

        @testset "q = $(q) | $(T) | exp_and_gram_chol" begin
            for t in ts
                Φgt, Ggt = mf_exp_and_gram(A, B, t)
                Φ, U = exp_and_gram_chol(A, B, t, method)
                G = U' * U
                FHG._symmetrize!(G)
                @test isapprox(err(Φ, Φgt), zT, atol = tols[i])
                @test isapprox(err(G, Ggt), zT, atol = tols[i])
            end
            Φgt, Ggt = mf_exp_and_gram(A, B)
            Φ, U = exp_and_gram_chol(A, B, method)
            G = U' * U
            FHG._symmetrize!(G)
            @test isapprox(err(G, Ggt), zT, atol = tols[i])
        end

        @testset "q = $(q) | on-square initial Gramian" begin
            # test covering the case when initial cholesky factor is non-square
            m = 1
            n = m * (q + 2)
            A = -tril(ones(T, n, n))
            B = ones(T, n, m)
            @test_nowarn exp_and_gram_chol(A, B, method)
        end

    end

end
