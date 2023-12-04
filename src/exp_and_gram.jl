"""
    ExpAndGram{T,N,A,B,C} <: AbstractExpAndGramAlgorithm

Non-adaptive algorithm of order N for computing the matrix exponential and an associated Gramian.

Constructor:

ExpAndGram{T,N}()

creates an algorithm with coefficients stored in the numeric type T of order N.
Current supported values of N are 3, 5, 7, 9, 13.

"""
struct ExpAndGram{T,N,A,B} <: AbstractExpAndGramAlgorithm where {T,N,A,B}
    pade_num::A
    gram_coeffs::B
    normtol::T
end

function exp_and_gram(
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
    method::AbstractExpAndGramAlgorithm,
) where {T<:Number}
    return exp_and_gram!(similar(A), similar(A), copy(A), copy(B), method)
end

function exp_and_gram(
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
    t::Number,
    method::AbstractExpAndGramAlgorithm,
) where {T<:Number}
    return exp_and_gram!(similar(A), similar(A), copy(A), copy(B), t, method)
end

function alloc_mem(A, B, ::ExpAndGram{T,q}) where {T,q}
    n, m = size(B)
    ncoeffhalf = div(q + 1, 2)
    return (
        pre_array = similar(A, 2n, n),
        tmp = similar(A),
        _A = similar(A),
        _B = similar(B),
        P = similar(A),
        A2 = similar(A),
        odd = similar(A),
        even = similar(A),
        tmpA1 = similar(A),
        tmpA2 = similar(A),
        tmpA3 = similar(A),
        L = zeros(eltype(A), n, m * (q + 1)),
        Loddtmp = similar(B, n, m*ncoeffhalf),
    )
end
function alloc_mem(A, B, method::ExpAndGram{T,13}) where {T}
    q = 13
    n, m = size(B)
    return (
        _A = similar(A),
        _B = similar(B),
        A2 = similar(A),
        A4 = similar(A),
        A6 = similar(A),
        tmpA1 = similar(A),
        tmpA2 = similar(A),
        tmpA3 = similar(A),
        L = similar(A, n, m * (q + 1)),
        tmpB2 = similar(B),
        A2B = similar(B),
        A4B = similar(B),
        A6B = similar(B),
        pre_array = similar(A, 2n, n),
        tmp = similar(A),
    )
end

function exp_and_gram!(
    eA::AbstractMatrix{T},
    U::AbstractMatrix{T},
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
    method::AbstractExpAndGramAlgorithm,
    cache = alloc_mem(A, B, method),
) where {T<:Number}
    Φ, U = exp_and_gram_chol!(eA, U, A, B, method, cache)
    G = isnothing(cache) ? copy(U) : cache._A
    mul!(G, U', U)
    _symmetrize!(G)
    return Φ, G
end

function exp_and_gram!(
    eA::AbstractMatrix{T},
    U::AbstractMatrix{T},
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
    t::Number,
    method::AbstractExpAndGramAlgorithm,
    cache = alloc_mem(A, B, method),
) where {T<:Number}
    Φ, U = exp_and_gram_chol!(eA, U, A, B, t, method, cache)
    G = isnothing(cache) ? copy(U) : cache._A
    mul!(G, U', U)
    _symmetrize!(G)
    return Φ, G
end



exp_and_gram_chol(
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
    method::AbstractExpAndGramAlgorithm,
) where {T<:Number} =
    exp_and_gram_chol!(similar(A), similar(A), copy(A), copy(B), method)

exp_and_gram_chol(
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
    t::Number,
    method::AbstractExpAndGramAlgorithm,
) where {T<:Number} =
    exp_and_gram_chol!(similar(A), similar(A), copy(A), copy(B), t, method)


"""
exp_and_gram_chol!(
    eA::AbstractMatrix{T},
    U::AbstractMatrix{T},
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
    method::AbstractExpAndGramAlgorithm,
    cache = alloc_mem(A, B, method),
)

Computes the matrix exponential of A * t and the controllability Gramian of (A, B) on the interval [0, 1].
The result is stored in (eA, _U), which are returned.
"""
function exp_and_gram_chol!(
    eA::AbstractMatrix{T},
    U::AbstractMatrix{T},
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
    method::AbstractExpAndGramAlgorithm,
    cache = alloc_mem(A, B, method),
) where {T<:Number}
    return exp_and_gram_chol!(eA, U, A, B, true, method, cache)
end

"""
exp_and_gram_chol!(
    eA::AbstractMatrix{T},
    U::AbstractMatrix{T},
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
    [t::Number],
    method::ExpAndGram{T,q},
    [cache = alloc_mem(A, B, method)],
)

Computes the matrix exponential of A * t and the controllability Gramian of (A, B) on the interval [0, t].
The result is stored in (eA, U), which are returned.
"""
function exp_and_gram_chol!(
    eA::AbstractMatrix{T},
    U::AbstractMatrix{T},
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
    t::Number,
    method::ExpAndGram{T,q},
    cache = alloc_mem(A, B, method),
) where {T<:Number,q}
    At, Bt = if cache == nothing
        (A * t, B * sqrt(t))
    else
        (mul!(cache._A, A, t), mul!(cache._B, B, sqrt(t)))
    end

    n, m = _dims_if_compatible(A::AbstractMatrix, B::AbstractMatrix) # first checks that (A, B) have compatible dimensions
    normA = opnorm(At, 1)
    sexp = log2(normA / method.normtol) # power required for accuracy of exp
    sgram = n <= q + 1 ? 0 : ceil(Int, log2((n - 1) / q)) # power requried for rank equivalence
    s = max(sexp, sgram)

    if s > 0
        si = ceil(Int, s)
        At ./= convert(T, 2^si)
        Bt ./= convert(T, sqrt(2^si))
    end

    # eA, U contains initial approximations, both are square
    eA, U = _exp_and_gram_chol_init!(eA, U, At, Bt, method, cache)

    if s > 0
        # eA, U contain final approximations
        eA, U = _exp_and_gram_double!(eA, U, si, cache)
    end

    # fix the diagonal signs on U
    triu2cholesky_factor!(U)
    return eA, U
end


function _exp_and_gram_double!(eA, U, s, cache)
    n = LinearAlgebra.checksquare(U)
    if isnothing(cache)
        cache = (pre_array = similar(U, 2n, n), tmp = similar(Φ0))
    end
    @unpack pre_array = cache

    for _ = 1:s
        # form pre-array for doubling of U
        mul!(view(pre_array, 1:n, 1:n), U, eA')
        pre_array[n+1:2n, 1:n] .= U

        # doubling of eA (use U as an intermediate array)
        mul!(U, eA, eA)
        eA .= U

        # form doubled U in square, triangular form
        U .= qr!(pre_array).R
    end
    return eA, U
end



"""
    _exp_and_gram_init(A::AbstractMatrix{T}, B::AbstractMatrix{T}, L::LegendreExp{T})

Computes the matrix exponential exp(A) and the controllability Grammian ∫_0^1 exp(A*t)*B*B'*exp(A'*t) dt,
using a Legendre expansion of the matrix exponential.
"""
function _exp_and_gram_chol_init!(
    eA::AbstractMatrix{T},
    U::AbstractMatrix{T},
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
    method::ExpAndGram{T,q},
    cache = alloc_mem(A, B, method),
) where {T,q}
    @unpack P, A2, L, tmpA1, tmpA2, tmpA3, odd, even, Loddtmp = cache

    n, m = _dims_if_compatible(A::AbstractMatrix, B::AbstractMatrix) # first checks that (A, B) have compatible dimensions
    isodd(q) || throw(DomainError(q, "The degree $(q) must be odd")) # code heavily assumes odd degree expansion

    # fetch expansion coefficients
    pade_num = method.pade_num
    gram_coeffs = method.gram_coeffs
    ncoeffhalf = div(q + 1, 2)

    mul!(A2, A, A)
    copy!(P, A2)

    # odd = pade_num[2] * I + pade_num[4] * P
    mul!(odd, pade_num[2], I)
    mul!(odd, pade_num[4], P, true, true)
    # even = pade_num[1] * I + pade_num[3] * P
    mul!(even, pade_num[1], I)
    mul!(even, pade_num[3], P, true, true)


    Leven = view(L, :, 1:m*ncoeffhalf)
    Lodd = view(L, :, m*ncoeffhalf+1:m*2*ncoeffhalf)

    # initialize zeroth block
    L0 = view(Leven, 1:n, 1:m)
    L0 .= B
    mul!(L0, P, B, gram_coeffs[1, 3], gram_coeffs[1, 1])

    # initialize first block (contains only A if deg < 4, else A and A^3)
    L1 = view(Lodd, 1:n, 1:m)
    L1 .= B
    mul!(L1, P, B, q < 4 ? false : gram_coeffs[2, 4], gram_coeffs[2, 2])

    # initialize second block (contains only A^2)
    L2 = view(Leven, 1:n, m+1:2m)
    mul!(L2, P, B, gram_coeffs[3, 3], true)

    # initialize third block (contains only A^3)
    L3 = view(Lodd, 1:n, m+1:2m)
    mul!(L3, P, B, gram_coeffs[4, 4], true)

    for k = 2:(div(length(pade_num), 2)-1)
        mul!(tmpA1, P, A2)
        copy!(P, tmpA1)
        mul!(even, pade_num[2k+1], P, true, true)
        mul!(odd, pade_num[2k+2], P, true, true)

        for i = 0:div(q - 1, 2)
            Leveni = view(Leven, 1:n, i*m+1:(i+1)*m)
            gram_coeffs[2i+1, 2k+1] != 0 &&
                mul!(Leveni, P, B, gram_coeffs[2i+1, 2k+1], true)
            Loddi = view(Lodd, 1:n, i*m+1:(i+1)*m)
            gram_coeffs[2i+2, 2k+2] != 0 && mul!(Loddi, P, B, gram_coeffs[2i+2, 2k+2], true)
        end
    end

    odd = mul!(tmpA1, A, odd)
    den = tmpA2 .= even .- odd # pade denominator
    num = tmpA3 .= even .+ odd # pade numerator

    Lodd .= mul!(Loddtmp, A, Lodd) # writes into L as Lodd and L share memory

    F = lu!(den)
    expA = num
    ldiv!(F, expA)
    ldiv!(F, L)

    _U = qr!(L').R # right Cholesky factor of the Grammian (may not be square!!)
    _U = triu2cholesky_factor!(_U)

    copy!(eA, expA)
    U .= 0
    copy!(view(U, 1:size(_U, 1), 1:size(_U, 2)), _U)
    return eA, U
end


function _exp_and_gram_chol_init!(
    eA::AbstractMatrix{T},
    U::AbstractMatrix{T},
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
    method::ExpAndGram{T,13},
    cache = alloc_mem(A, B, method),
) where {T}
    @unpack A2, A4, A6, tmpA1, tmpA2, tmpA3, L, tmpB2, A2B, A4B, A6B = cache

    n, m = size(B)
    n == LinearAlgebra.checksquare(A) || throw(
        DimensionMismatch(
            "size of A, $(LinearAlgebra.checksquare(A)), incompatible with the size of B, $(size(B)).",
        ),
    )

    # fetch expansion coefficients
    pade_num = method.pade_num
    gram_coeffs = method.gram_coeffs
    q = 13

    mul!(A2, A, A)
    mul!(A4, A2, A2)
    mul!(A6, A2, A4)

    @. tmpA1 = pade_num[14] * A6 + pade_num[12] * A4 + pade_num[10] * A2
    @. tmpA2 = pade_num[8] * A6 + pade_num[6] * A4 + pade_num[4] * A2
    mul!(tmpA2, true, pade_num[2] * I, true, true)
    _U = mul!(tmpA2, A6, tmpA1, true, true)
    _U = mul!(tmpA1, A, _U) # _U is odd terms

    @. tmpA3 = pade_num[13] * A6 + pade_num[11] * A4 + pade_num[9] * A2
    @. tmpA2 = pade_num[7] * A6 + pade_num[5] * A4 + pade_num[3] * A2
    mul!(tmpA2, true, pade_num[1] * I, true, true)
    V = mul!(tmpA2, A6, tmpA3, true, true) # V is even terms

    @. tmpA3 = V + _U # numerator
    @. tmpA2 = V - _U # denominator
    num = tmpA3
    den = tmpA2

    mul!(A2B, A2, B)
    mul!(A4B, A4, B)
    mul!(A6B, A6, B)

    # L0
    L0 = view(L, 1:n, 1:m)
    @. L0 = gram_coeffs[1, 3] * A2B + gram_coeffs[1, 5] * A4B + gram_coeffs[1, 7] * A6B # low order terms
    mul!(L0, gram_coeffs[1, 1] * I, B, true, true) # add constant term in A
    @. tmpB2 = gram_coeffs[1, 9] * A2B + gram_coeffs[1, 11] * A4B + gram_coeffs[1, 13] * A6B # high order terms bar factor 6
    mul!(L0, A6, tmpB2, true, true)

    # L2
    L2 = view(L, 1:n, 2m+1:3m)
    @. L2 = gram_coeffs[3, 3] * A2B + gram_coeffs[3, 5] * A4B + gram_coeffs[3, 7] * A6B # low order terms
    @. tmpB2 = gram_coeffs[3, 9] * A2B + gram_coeffs[3, 11] * A4B + gram_coeffs[3, 13] * A6B # high order terms bar factor 6
    mul!(L2, A6, tmpB2, true, true)

    # L4
    L4 = view(L, 1:n, 4m+1:5m)
    @. L4 = gram_coeffs[5, 5] * A4B + gram_coeffs[5, 7] * A6B # low order terms
    @. tmpB2 = gram_coeffs[5, 9] * A2B + gram_coeffs[5, 11] * A4B + gram_coeffs[5, 13] * A6B # high order terms bar factor 6
    mul!(L4, A6, tmpB2, true, true)

    # L6
    L6 = view(L, 1:n, 6m+1:7m)
    @. L6 = gram_coeffs[7, 7] * A6B # low order terms
    @. tmpB2 = gram_coeffs[7, 9] * A2B + gram_coeffs[7, 11] * A4B + gram_coeffs[7, 13] * A6B # high order terms bar factor 6
    mul!(L6, A6, tmpB2, true, true)

    # L8
    L8 = view(L, 1:n, 8m+1:9m)
    @. tmpB2 = gram_coeffs[9, 9] * A2B + gram_coeffs[9, 11] * A4B + gram_coeffs[9, 13] * A6B # high order terms bar factor 6
    mul!(L8, A6, tmpB2, true, false)

    # L10
    L10 = view(L, 1:n, 10m+1:11m)
    @. tmpB2 = gram_coeffs[11, 11] * A4B + gram_coeffs[11, 13] * A6B # high order terms bar factor 6
    mul!(L10, A6, tmpB2, true, false)

    # L12
    L12 = view(L, 1:n, 12m+1:13m)
    @. tmpB2 = gram_coeffs[11, 13] * A6B # high order terms bar factor 6
    mul!(L12, A6, tmpB2, true, false)

    # L1
    L1 = view(L, 1:n, m+1:2m)
    @. L1 = gram_coeffs[2, 4] * A2B + gram_coeffs[2, 6] * A4B + gram_coeffs[2, 8] * A6B # low order terms
    mul!(L1, gram_coeffs[2, 2] * I, B, true, true) # add constant term in A
    @. tmpB2 = gram_coeffs[2, 10] * A2B + gram_coeffs[2, 12] * A4B # high order terms bar factor 6
    mul!(L1, A6, tmpB2, true, true)
    mul!(tmpB2, A, L1, true, false)
    copy!(L1, tmpB2)

    # L3
    L3 = view(L, 1:n, 3m+1:4m)
    @. L3 = gram_coeffs[4, 4] * A2B + gram_coeffs[4, 6] * A4B + gram_coeffs[4, 8] * A6B # low order terms
    @. tmpB2 = gram_coeffs[4, 10] * A2B + gram_coeffs[4, 12] * A4B # high order terms bar factor 6
    mul!(L3, A6, tmpB2, true, true)
    mul!(tmpB2, A, L3, true, false)
    copy!(L3, tmpB2)

    # L5
    L5 = view(L, 1:n, 5m+1:6m)
    @. L5 = gram_coeffs[6, 6] * A4B + gram_coeffs[6, 8] * A6B # low order terms
    @. tmpB2 = gram_coeffs[6, 10] * A2B + gram_coeffs[6, 12] * A4B # high order terms bar factor 6
    mul!(L5, A6, tmpB2, true, true)
    mul!(tmpB2, A, L5, true, false)
    copy!(L5, tmpB2)

    # L7
    L7 = view(L, 1:n, 7m+1:8m)
    @. L7 = gram_coeffs[8, 8] * A6B # low order terms
    @. tmpB2 = gram_coeffs[8, 10] * A2B + gram_coeffs[8, 12] * A4B # high order terms bar factor 6
    mul!(L7, A6, tmpB2, true, true)
    mul!(tmpB2, A, L7, true, false)
    copy!(L7, tmpB2)

    # L9
    L9 = view(L, 1:n, 9m+1:10m)
    @. tmpB2 = gram_coeffs[10, 10] * A2B + gram_coeffs[10, 12] * A4B # high order terms bar factor 6
    mul!(L9, A6, tmpB2, true, false)
    mul!(tmpB2, A, L9, true, false)
    copy!(L9, tmpB2)

    # L11
    L11 = view(L, 1:n, 11m+1:12m)
    @. tmpB2 = gram_coeffs[12, 12] * A4B # high order terms bar factor 6
    mul!(L11, A6, tmpB2, true, false)
    mul!(tmpB2, A, L11, true, false)
    copy!(L11, tmpB2)

    # L13
    L13 = view(L, 1:n, 13m+1:14m)
    @. tmpB2 = gram_coeffs[14, 14] * A6B # high order terms bar factor 6
    mul!(L13, A6, tmpB2, true, false)
    mul!(tmpB2, A, L13, true, false)
    copy!(L13, tmpB2)

    F = lu!(den)
    expA = num
    ldiv!(F, expA)
    ldiv!(F, L)

    _U = qr!(L').R # right Cholesky factor of the Grammian (may not be square!!)
    _U = triu2cholesky_factor!(_U)

    copy!(eA, expA)
    U .= 0
    copy!(view(U, 1:size(_U, 1), 1:size(_U, 2)), _U)
    return eA, U
end


# constructors

function ExpAndGram{T,3}() where {T}
    pade_num = T[120, 60, 12, 1]
    gramcs = T[
        120.0 0.0 2.0 0.0
        0.0 34.64101615137755 0.0 0.0
        0.0 0.0 4.47213595499958 0.0
        0.0 0.0 0.0 0.37796447300922725
    ]
    normtol = T(0.00067)
    ExpAndGram{T,3,typeof(pade_num),typeof(gramcs)}(pade_num, gramcs, normtol)
end

function ExpAndGram{T,5}() where {T}
    pade_num = T[30240, 15120, 3360, 420, 30, 1]
    gramcs = T[
        30240.0 0.0 840.0 0.0 2.0 0.0
        0.0 8729.536070147142 0.0 96.99484522385713 0.0 0.0
        0.0 0.0 1126.978260659894 0.0 4.47213595499958 0.0
        0.0 0.0 0.0 95.24704719832526 0.0 0.0
        0.0 0.0 0.0 0.0 6.0 0.0
        0.0 0.0 0.0 0.0 0.0 0.30151134457776363
    ]
    normtol = T(0.021)
    ExpAndGram{T,5,typeof(pade_num),typeof(gramcs)}(pade_num, gramcs, normtol)
end

function ExpAndGram{T,7}() where {T}
    pade_num = T[17297280, 8648640, 1995840, 277200, 25200, 1512, 56, 1]
    gramcs = T[
        1.729728e7 0.0 554400.0 0.0 3024.0 0.0 2.0 0.0
        0.0 4.993294632124165e6 0.0 76819.91741729484 0.0 187.06148721743875 0.0 0.0
        0.0 0.0 644631.5650974594 0.0 5312.897514539501 0.0 4.47213595499958 0.0
        0.0 0.0 0.0 54481.31099744205 0.0 232.82611537368396 0.0 0.0
        0.0 0.0 0.0 0.0 3432.0 0.0 6.0 0.0
        0.0 0.0 0.0 0.0 0.0 172.46448909848078 0.0 0.0
        0.0 0.0 0.0 0.0 0.0 0.0 7.211102550927978 0.0
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.25819888974716115
    ]
    normtol = T(0.13)
    ExpAndGram{T,7,typeof(pade_num),typeof(gramcs)}(pade_num, gramcs, normtol)
end

function ExpAndGram{T,9}() where {T}
    pade_num = T[
        17643225600,
        8821612800,
        2075673600,
        302702400,
        30270240,
        2162160,
        110880,
        3960,
        90,
        1,
    ]
    gramcs = T[
        1.76432256e10 0.0 6.054048e8 0.0 4.32432e6 0.0 7920.0 0.0 2.0 0.0
        0.0 5.093160524766648e9 0.0 8.987930337823497e7 0.0 356663.9022945832 0.0 304.8409421321224 0.0 0.0
        0.0 0.0 6.575241963994086e8 0.0 6.90676676890135e6 0.0 15348.370597558556 0.0 4.47213595499958 0.0
        0.0 0.0 0.0 5.557093721739089e7 0.0 363208.739982947 0.0 412.73720452607614 0.0 0.0
        0.0 0.0 0.0 0.0 3.50064e6 0.0 14040.0 0.0 6.0 0.0
        0.0 0.0 0.0 0.0 0.0 175913.7788804504 0.0 397.994974842648 0.0 0.0
        0.0 0.0 0.0 0.0 0.0 0.0 7355.324601946538 0.0 7.211102550927978 0.0
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 263.36286754210437 0.0 0.0
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 8.246211251235321 0.0
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.22941573387056177
    ]
    normtol = T(0.41)
    ExpAndGram{T,9,typeof(pade_num),typeof(gramcs)}(pade_num, gramcs, normtol)
end

function ExpAndGram{T,13}() where {T}
    pade_num = T[
        64764752532480000,
        32382376266240000,
        7771770303897600,
        1187353796428800,
        129060195264000,
        10559470521600,
        670442572800,
        33522128640,
        1323241920,
        40840800,
        960960,
        16380,
        182,
        1,
    ]
    gramcs = T[
        6.476475253248e16 0.0 2.3747075928576e15 0.0 2.11189410432e13 0.0 6.704425728e10 0.0 8.16816e7 0.0 32760.0 0.0 2.0 0.0
        0.0 1.8695973654313412e16 0.0 3.7391947308626825e14 0.0 2.0902330793642324e12 0.0 4.300891109802947e9 0.0 3.2153791191708636e6 0.0 623.5382907247958 0.0 0.0
        0.0 0.0 2.413639820142949e15 0.0 3.103251197326649e13 0.0 1.1660095749482845e11 0.0 1.5773939055036315e8 0.0 68423.68011149357 0.0 4.47213595499958 0.0
        0.0 0.0 0.0 2.0398979633759847e14 0.0 1.8132426341119866e12 0.0 4.703322049944629e9 0.0 4.101972832674541e6 0.0 899.5554457619608 0.0 0.0
        0.0 0.0 0.0 0.0 1.2850149312e13 0.0 8.177367744e10 0.0 1.465128e8 0.0 77520.0 0.0 6.0 0.0
        0.0 0.0 0.0 0.0 0.0 6.457442995143573e11 0.0 2.98035830545088e9 0.0 3.5994665524769085e6 0.0 1008.2539362680416 0.0 0.0
        0.0 0.0 0.0 0.0 0.0 0.0 2.699992554882535e10 0.0 8.999975182941784e7 0.0 69053.51802768632 0.0 7.211102550927978 0.0
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 9.667524141735567e8 0.0 2.2747115627613096e6 0.0 975.991803244269 0.0 0.0
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0270192261034615e7 0.0 47795.04041215992 0.0 8.246211251235321 0.0
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 842139.2758920582 0.0 802.0374056114839 0.0 0.0
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 21079.848196796866 0.0 9.16515138991168 0.0
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 479.58315233127195 0.0 0.0
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 10.0 0.0
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.19245008972987526
    ]
    normtol = T(1.57)
    ExpAndGram{T,13,typeof(pade_num),typeof(gramcs)}(pade_num, gramcs, normtol)
end
