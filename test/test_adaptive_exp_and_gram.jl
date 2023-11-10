function test_adaptive_exp_and_gram(T)

    n, m = 8, 2
    A = randn(T, n, n)
    B = randn(T, n, m)

    tol = 5e-14

    ts = [0.5, 1.0, 1.5]

    method = AdaptiveExpAndGram{T}()

    err(G1, G2) = opnorm(G1 - G2, 1) / opnorm(G2, 1)

    for t in ts
        Φgt, Ggt = mf_exp_and_gram(A, B, t)
        Φ, G = exp_and_gram(A, B, t, method)
        @test isapprox(err(Φ, Φgt), zero(T), atol = tol)
        @test isapprox(err(G, Ggt), zero(T), atol = tol)
    end
    Φgt, Ggt = mf_exp_and_gram(A, B)
    Φ, G = exp_and_gram(A, B, method)
    @test isapprox(err(Φ, Φgt), zero(T), atol = tol)
    @test isapprox(err(G, Ggt), zero(T), atol = tol)

    # make sure to trigger all branches
    A = tril(ones(T, n, n)) / n
    B = ones(T, n, 1)

    normtols = T[0.00067, 0.021, 0.13, 0.41, 1.57]
    scalings = vcat(normtols / 2, T(2))
    for s in scalings
        As = A * s
        Bs = B * sqrt(s)
        Φgt, Ggt = mf_exp_and_gram(As, Bs)
        Φ, G = exp_and_gram(As, Bs, method)
        @test isapprox(err(Φ, Φgt), zero(T), atol = tol)
        @test isapprox(err(G, Ggt), zero(T), atol = tol)
    end

    # test covering the case when initial cholesky factor is non-square
    q = 13
    m = 1 
    n = m * (q + 2) 
    A = - tril(ones(n, n))
    B = ones(n, m)
    @test_nowarn exp_and_gram(A, B, method)

end
