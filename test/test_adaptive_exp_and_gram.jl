function test_adaptive_exp_and_gram(T)

    n, m = 8, 2
    L = randn(T, n, n)
    Z = randn(T, n, n)
    A = - L*L'/2 + (Z - Z')/2
    B = randn(T, n, m)

    RT = real(T)
    if RT <: Float32
        tol = 5f-6
    else
        tol = 5e-14
    end

    zT = zero(RT)

    ts = [0.5, 1.0, 1.5]

    method = AdaptiveExpAndGram{T}()

    err(G1, G2) = opnorm(G1 - G2, 1) / opnorm(G2, 1)

    for t in ts
        Φgt, Ggt = mf_exp_and_gram(A, B, t)
        Φ, G = exp_and_gram(A, B, t, method)
        @test isapprox(err(Φ, Φgt), zT, atol = tol)
        @test isapprox(err(G, Ggt), zT, atol = tol)
        @test ishermitian(G)
    end
    Φgt, Ggt = mf_exp_and_gram(A, B)
    Φ, G = exp_and_gram(A, B, method)
    @test isapprox(err(Φ, Φgt), zT, atol = tol)
    @test isapprox(err(G, Ggt), zT, atol = tol)

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
        @test isapprox(err(Φ, Φgt), zT, atol = tol)
        @test isapprox(err(G, Ggt), zT, atol = tol)
    end

    # test covering the case when initial cholesky factor is non-square
    q = 13
    m = 1
    n = m * (q + 2)
    A = -tril(ones(T, n, n))
    B = ones(T, n, m)
    @test_nowarn exp_and_gram(A, B, method)

end
