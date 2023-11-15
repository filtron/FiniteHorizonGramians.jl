function test_exp_and_gram(T)

    n, m = 8, 2
    A = randn(T, n, n)
    B = randn(T, n, m)

    tol = 1e-11

    ts = [0.5, 1.0, 1.5]

    qs = [3, 5, 7, 9, 13]

    for q in qs

        method = ExpAndGram{T,q}()

        err(G1, G2) = opnorm(G1 - G2, 1) / opnorm(G2, 1)

        @testset "q = $(q) | exp_and_gram" begin
            for t in ts
                Φgt, Ggt = mf_exp_and_gram(A, B, t)
                Φ, G = exp_and_gram(A, B, t, method)
                @test isapprox(err(Φ, Φgt), zero(T), atol = tol)
                @test isapprox(err(G, Ggt), zero(T), atol = tol)
            end
            Φgt, Ggt = mf_exp_and_gram(A, B)
            Φ, G = exp_and_gram(A, B, method)
            @test isapprox(err(G, Ggt), zero(T), atol = tol)
        end

        @testset "q = $(q) | exp_and_gram_chol" begin
            for t in ts
                Φgt, Ggt = mf_exp_and_gram(A, B, t)
                Φ, U = exp_and_gram_chol(A, B, t, method)
                G = U'*U
                FHG._symmetrize!(G)
                @test isapprox(err(Φ, Φgt), zero(T), atol = tol)
                @test isapprox(err(G, Ggt), zero(T), atol = tol)
            end
            Φgt, Ggt = mf_exp_and_gram(A, B)
            Φ, U = exp_and_gram_chol(A, B, method)
            G = U'*U
            FHG._symmetrize!(G)
            @test isapprox(err(G, Ggt), zero(T), atol = tol)
        end

        @testset "q = $(q) | on-square initial Gramian" begin
            # test covering the case when initial cholesky factor is non-square
            m = 1
            n = m * (q + 2)
            A = - tril(ones(n, n))
            B = ones(n, m)
            @test_nowarn exp_and_gram_chol(A, B, method)
        end

    end

end
