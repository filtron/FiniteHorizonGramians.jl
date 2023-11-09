
function integrator2AB(T, ndiff)
    A = diagm(-1 => ones(T, ndiff))
    B = vcat(one(T), zeros(T, ndiff, 1))
    return A, B
end

function integrator_exp_and_gram_chol(T, t, ndiff)

end

function integrator_exp_and_gram_chol(T, ndiff)
    # transition matrix
    irange = collect(0:ndiff)
    g = @. exp(-logfactorial(irange))
    Φ = zeros(T, ndiff + one(ndiff), ndiff + one(ndiff))
    @simd ivdep for col = 0:ndiff
        @simd ivdep for row = col:ndiff
            @inbounds Φ[row+1, col+1] = g[row-col+1]
        end
    end

    # cholesky factor
    L = zeros(T, ndiff + one(ndiff), ndiff + one(ndiff))
    @simd ivdep for n = 0:ndiff
        something = logfactorial(n)
        @simd ivdep for m = 0:ndiff
            if m <= n
                @inbounds L[n+1, m+1] = exp(
                    something + T(0.5) * log(2 * m + 1) - logfactorial(n - m) -
                    logfactorial(n + m + 1),
                )
            end
        end
    end
    U = permutedims(L)
    return Φ, U
end

function integrator_exp_and_gram(T, ndiff)
    Φ, U = integrator_exp_and_gram_chol(T, ndiff)
    G = U' * U
    FHG._symmetrize!(G)
    return Φ, G
end


function mf_exp_and_gram(A, B)
    n = LinearAlgebra.checksquare(A)
    pre_array = [A B*B'; zero(A) -A']
    post_array = exp(pre_array)
    Φ = post_array[1:n, 1:n]
    G = post_array[1:n, n+1:2n] * Φ'
    FHG._symmetrize!(G)
    Φ, G
end
