numeric_types = (Float32, Float64, ComplexF32, ComplexF64)
n = 5

for T in numeric_types
    A = (I - T(2) * tril(ones(T, n, n)))
    B = sqrt(T(2)) * ones(T, n, 1)
    Bvec = vec(B)
    R = real(T)

    @test gramcond(A, B) ≈
          gramcond(A, Bvec) ≈
          R(2) * (one(R) + one(R) / opnorm(B, 2)) * (opnorm(A, 2) + opnorm(B, 2))
end
