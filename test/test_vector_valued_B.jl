function test_vector_valued_B(T)

    n = 5
    A = (I - T(2) * tril(ones(T, n, n)))
    B = sqrt(T(2)) * ones(T, n, 1)
    Bvec = vec(B)
    t = one(T)

    algs = [ExpAndGram{T,q}() for q in [3, 5, 7, 9, 13]]
    algs = vcat(algs, AdaptiveExpAndGram{T}())

    for alg in algs

        # non-mutating gram
        @test all(exp_and_gram(A, B, alg) .≈ exp_and_gram(A, Bvec, alg))
        @test all(exp_and_gram(A, B, t, alg) .≈ exp_and_gram(A, Bvec, t, alg))

        # mutating gram
        @test all(
            exp_and_gram!(similar(A), similar(A), A, B, alg) .≈
            exp_and_gram!(similar(A), similar(A), A, Bvec, alg),
        )
        @test all(
            exp_and_gram!(similar(A), similar(A), A, B, t, alg) .≈
            exp_and_gram!(similar(A), similar(A), A, Bvec, t, alg),
        )

        # non-mutating chol
        @test all(exp_and_gram_chol(A, B, alg) .≈ exp_and_gram_chol(A, Bvec, alg))
        @test all(exp_and_gram_chol(A, B, t, alg) .≈ exp_and_gram_chol(A, Bvec, t, alg))

        # mutating chol
        @test all(
            exp_and_gram_chol!(similar(A), similar(A), A, B, alg) .≈
            exp_and_gram_chol!(similar(A), similar(A), A, Bvec, alg),
        )
        @test all(
            exp_and_gram_chol!(similar(A), similar(A), A, B, t, alg) .≈
            exp_and_gram_chol!(similar(A), similar(A), A, Bvec, t, alg),
        )

    end


end
