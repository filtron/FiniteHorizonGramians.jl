
function integrator2AB(T, ndiff)
    A = diagm(-1 => ones(T, ndiff))
    B = vcat(one(T), zeros(T, ndiff, 1))
    return A, B
end


integrator_exp_and_gram_chol(T, ndiff) = integrator_exp_and_gram_chol(T, true, ndiff)

function integrator_exp_and_gram_chol(T, t, ndiff)
    # transition matrix
    irange = collect(0:ndiff)
    g = @. exp(-logfactorial(irange)) * t^irange
    Φ = zeros(T, ndiff + one(ndiff), ndiff + one(ndiff))
    @simd ivdep for col = 0:ndiff
        @simd ivdep for row = col:ndiff
            @inbounds Φ[row+1, col+1] = g[row-col+1]
        end
    end

    # cholesky factor
    L = zeros(T, ndiff + one(ndiff), ndiff + one(ndiff))
    @simd ivdep for n = 0:ndiff
        something = logfactorial(n) * sqrt(t) * t^n
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

integrator_exp_and_gram(T, ndiff) = integrator_exp_and_gram(T, true, ndiff)

function integrator_exp_and_gram(T, t, ndiff)
    Φ, U = integrator_exp_and_gram_chol(T, t, ndiff)
    G = U' * U
    FHG._symmetrize!(G)
    return Φ, G
end

mf_exp_and_gram(A, B) = mf_exp_and_gram(A, B, true)

function mf_exp_and_gram(A, B, t)
    A, B, t = big.(A), big.(B), big.(t)
    n = LinearAlgebra.checksquare(A)

    pre_array = [A B*B'; zero(A) -A']

    method = ExpMethodGeneric()
    cache = ExponentialUtilities.alloc_mem(pre_array, method)
    #post_array = exp(pre_array * t)
    post_array = exponential!(pre_array .* t, method, cache)
    Φ = post_array[1:n, 1:n]
    G = post_array[1:n, n+1:2n] * Φ'
    FHG._symmetrize!(G)
    Φ, G
end
