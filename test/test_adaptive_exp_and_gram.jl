function test_adaptive_exp_and_gram(T)

    n, m = 8, 2
    A = randn(T, n, n)
    B = randn(T, n, m)

    tol = 5e-14

    nsample = 25
    method = AdaptiveExpAndGram{T}()

    err(G1, G2) = opnorm(G1 - G2, 1) / opnorm(G2, 1)

    relerrs_Φ = zeros(T, nsample)
    relerrs_G = zeros(T, nsample)
    for i in eachindex(relerrs_Φ)
        Φgt, Ggt = mf_exp_and_gram(A, B)
        Φ, G = exp_and_gram(A, B, method)
        relerrs_Φ[i] = err(Φ, Φgt)
        relerrs_G[i] = err(G, Ggt)
    end

    for (relerr_Φ, relerr_G) in zip(relerrs_Φ, relerrs_G)
        @test isapprox(relerr_Φ, zero(T), atol = tol)
        @test isapprox(relerr_G, zero(T), atol = tol)
    end

end
