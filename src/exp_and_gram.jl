"""
    ExpAndGram{T,N,A,B,C} <: AbstractExpAndGramAlgorithm

Non-adaptive algorithm of order N for computing the matrix exponential and an associated Gramian. 

Constructor: 

ExpAndGram{T,N}()

creates an algorithm with coefficients stored in the numeric type T of order N. 

"""
struct ExpAndGram{T,N,A,B,C} <: AbstractExpAndGramAlgorithm where {T,N,A,B,C}
    pade_num::A
    leg_nums::B
    sqr_norms::C
    normtol::T
end

function exp_and_gram(
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
    method::ExpAndGram,
) where {T<:Number}
    Φ, G = exp_and_gram!(copy(A), copy(B), method)
    return Φ, G
end

function exp_and_gram!(
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
    method::ExpAndGram,
) where {T<:Number}
    Φ, U = exp_and_gram_chol!(A, B, method)
    G = U' * U
    _symmetrize!(G)
    return Φ, G
end

exp_and_gram_chol(
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
    method::ExpAndGram{T},
) where {T<:Number} = exp_and_gram_chol!(copy(A), copy(B), method)

function exp_and_gram_chol!(
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
    method::ExpAndGram{T,q},
) where {T<:Number,q}

    n, m = _dims_if_compatible(A::AbstractMatrix, B::AbstractMatrix) # first checks that (A, B) have compatible dimensions
    normA = opnorm(A, 1)
    sexp = log2(normA / method.normtol) # power required for accuracy of exp 
    sgram = n <= q + 1 ? 0 : ceil(Int, log2((n - 1) / q)) # power requried for rank equivalence
    s = max(sexp, sgram)

    if s > 0
        si = ceil(Int, s)
        A ./= convert(T, 2^si) # this mutates A
        B ./= convert(T, sqrt(2^si))
    end

    Φ, U = _exp_and_gram_chol_init(A, B, method)

    # should pre-allocate here 
    if s > 0
        Φ, U = _exp_and_gram_double(Φ, U, si)
    end

    triu2cholesky_factor!(U)
    return Φ, U
end


function _gram_coeffs(method::ExpAndGram)
    (; pade_num, leg_nums, sqr_norms) = method
    gram_coeffs = copy(leg_nums)
    for row in axes(gram_coeffs, 1)
        gram_coeffs[row, :] .= gram_coeffs[row, :] / sqrt(sqr_norms[row])
    end
    return gram_coeffs
end


function _exp_and_gram_double(Φ0, U0, s)
    Φ = Φ0
    m, n = size(U0)
    U = similar(Φ)
    U[1:m, 1:n] .= U0

    pre_array = similar(Φ, 2n, n)
    tmp = similar(Φ)
    for _ = 1:s
        sub_array = view(pre_array, 1:2m, 1:n)
        mul!(view(sub_array, 1:m, 1:n), view(U, 1:m, 1:n), Φ')
        sub_array[m+1:2m, 1:n] .= U[1:m, 1:n]
        m = min(n, 2 * m) # new row-size of U 
        U[1:m, 1:n] .= qr!(sub_array).R

        mul!(tmp, Φ, Φ)
        Φ .= tmp
    end
    return Φ, U
end



"""
    _exp_and_gram_init(A::AbstractMatrix{T}, B::AbstractMatrix{T}, L::LegendreExp{T})
    
Computes the matrix exponential exp(A) and the controllability Grammian ∫_0^1 exp(A*t)*B*B'*exp(A'*t) dt, 
using a Legendre expansion of the matrix exponential.   
"""
function _exp_and_gram_chol_init(
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
    method::ExpAndGram{T,q},
) where {T,q}

    n, m = _dims_if_compatible(A::AbstractMatrix, B::AbstractMatrix) # first checks that (A, B) have compatible dimensions
    isodd(q) || throw(DomainError(q, "The degree $(q) must be odd")) # code heavily assumes odd degree expansion 

    # fetch expansion coefficients 
    pade_num = method.pade_num
    gram_coeffs = _gram_coeffs(method)
    ncoeffhalf = div(q + 1, 2)

    A2 = A * A
    P = A2

    odd = mul!(pade_num[4] * P, true, pade_num[2] * I, true, true) # odd part of the pade numerator 
    even = mul!(pade_num[3] * P, true, pade_num[1] * I, true, true) # even part of the pade numerator 

    L = zeros(T, n, m * (q + 1)) # left square-root of the Grammian
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
        P *= A2
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

    odd = A * odd
    den = even - odd # pade denominator 
    num = even + odd # pade numerator

    Lodd .= A * Lodd

    F = lu!(den)
    expA = num
    ldiv!(F, expA)
    ldiv!(F, L)

    U = qr!(L').R # right Cholesky factor of the Grammian (may not be square!!)
    U = triu2cholesky_factor!(U)
    return expA, U
end


function _exp_and_gram_chol_init(
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
    method::ExpAndGram{T,13},
) where {T}

    n, m = size(B)
    n == LinearAlgebra.checksquare(A) || throw(
        DimensionMismatch(
            "size of A, $(LinearAlgebra.checksquare(A)), incompatible with the size of B, $(size(B)).",
        ),
    )

    # fetch expansion coefficients 
    pade_num = method.pade_num
    gram_coeffs = _gram_coeffs(method)
    q = 13

    A2 = A * A
    A4 = A2 * A2
    A6 = A2 * A4
    tmpA1, tmpA2 = similar(A6), similar(A6)

    tmpA1 .= pade_num[14] .* A6 .+ pade_num[12] .* A4 .+ pade_num[10] .* A2
    tmpA2 .= pade_num[8] .* A6 .+ pade_num[6] .* A4 .+ pade_num[4] .* A2
    mul!(tmpA2, true, pade_num[2] * I, true, true)
    U = mul!(tmpA2, A6, tmpA1, true, true)
    U = mul!(tmpA1, A, U) # U is odd terms 

    #tmpA1 = A # not good 
    tmpA1 = similar(A6) # quick fix 

    tmpA1 .= pade_num[13] .* A6 .+ pade_num[11] .* A4 .+ pade_num[9] .* A2
    tmpA2 .= pade_num[7] .* A6 .+ pade_num[5] .* A4 .+ pade_num[3] .* A2
    mul!(tmpA2, true, pade_num[1] * I, true, true)
    V = mul!(tmpA2, A6, tmpA1, true, true) # V is even terms 

    tmpA1 .= V .+ U # numerator 
    tmpA2 .= V .- U  # denominator 
    num = tmpA1
    den = tmpA2

    L = similar(A, n, m * (q + 1))

    A2B = A2 * B
    A4B = A4 * B
    A6B = A6 * B

    tmpB2 = similar(B)

    # L0 
    L0 = view(L, 1:n, 1:m)
    L0 .= gram_coeffs[1, 3] .* A2B + gram_coeffs[1, 5] .* A4B + gram_coeffs[1, 7] .* A6B # low order terms 
    mul!(L0, gram_coeffs[1, 1] * I, B, true, true) # add constant term in A 
    tmpB2 .=
        gram_coeffs[1, 9] .* A2B + gram_coeffs[1, 11] .* A4B + gram_coeffs[1, 13] .* A6B # high order terms bar factor 6 
    mul!(L0, A6, tmpB2, true, true)

    # L2 
    L2 = view(L, 1:n, 2m+1:3m)
    L2 .= gram_coeffs[3, 3] .* A2B + gram_coeffs[3, 5] .* A4B + gram_coeffs[3, 7] .* A6B # low order terms 
    tmpB2 .=
        gram_coeffs[3, 9] .* A2B + gram_coeffs[3, 11] .* A4B + gram_coeffs[3, 13] .* A6B # high order terms bar factor 6 
    mul!(L2, A6, tmpB2, true, true)

    # L4 
    L4 = view(L, 1:n, 4m+1:5m)
    L4 .= gram_coeffs[5, 5] .* A4B + gram_coeffs[5, 7] .* A6B # low order terms 
    tmpB2 .=
        gram_coeffs[5, 9] .* A2B + gram_coeffs[5, 11] .* A4B + gram_coeffs[5, 13] .* A6B # high order terms bar factor 6 
    mul!(L4, A6, tmpB2, true, true)

    # L6 
    L6 = view(L, 1:n, 6m+1:7m)
    L6 .= gram_coeffs[7, 7] .* A6B # low order terms 
    tmpB2 .=
        gram_coeffs[7, 9] .* A2B + gram_coeffs[7, 11] .* A4B + gram_coeffs[7, 13] .* A6B # high order terms bar factor 6 
    mul!(L6, A6, tmpB2, true, true)

    # L8 
    L8 = view(L, 1:n, 8m+1:9m)
    tmpB2 .=
        gram_coeffs[9, 9] .* A2B + gram_coeffs[9, 11] .* A4B + gram_coeffs[9, 13] .* A6B # high order terms bar factor 6 
    mul!(L8, A6, tmpB2, true, false)

    # L10 
    L10 = view(L, 1:n, 10m+1:11m)
    tmpB2 .= gram_coeffs[11, 11] .* A4B + gram_coeffs[11, 13] .* A6B # high order terms bar factor 6 
    mul!(L10, A6, tmpB2, true, false)

    # L12 
    L12 = view(L, 1:n, 12m+1:13m)
    tmpB2 .= gram_coeffs[11, 13] .* A6B # high order terms bar factor 6 
    L12 .= mul!(L12, A6, tmpB2, true, false)

    # L1 
    L1 = view(L, 1:n, m+1:2m)
    L1 .= gram_coeffs[2, 4] .* A2B + gram_coeffs[2, 6] .* A4B + gram_coeffs[2, 8] .* A6B # low order terms 
    mul!(L1, gram_coeffs[2, 2] * I, B, true, true) # add constant term in A 
    tmpB2 .= gram_coeffs[2, 10] .* A2B + gram_coeffs[2, 12] .* A4B # high order terms bar factor 6 
    mul!(L1, A6, tmpB2, true, true)
    mul!(tmpB2, A, L1, true, false)
    L1 .= tmpB2

    # L3 
    L3 = view(L, 1:n, 3m+1:4m)
    L3 .= gram_coeffs[4, 4] .* A2B + gram_coeffs[4, 6] .* A4B + gram_coeffs[4, 8] .* A6B # low order terms 
    tmpB2 .= gram_coeffs[4, 10] .* A2B + gram_coeffs[4, 12] .* A4B # high order terms bar factor 6 
    mul!(L3, A6, tmpB2, true, true)
    mul!(tmpB2, A, L3, true, false)
    L3 .= tmpB2

    # L5 
    L5 = view(L, 1:n, 5m+1:6m)
    L5 .= gram_coeffs[6, 6] .* A4B + gram_coeffs[6, 8] .* A6B # low order terms 
    tmpB2 .= gram_coeffs[6, 10] .* A2B + gram_coeffs[6, 12] .* A4B # high order terms bar factor 6 
    mul!(L5, A6, tmpB2, true, true)
    mul!(tmpB2, A, L5, true, false)
    L5 .= tmpB2

    # L7 
    L7 = view(L, 1:n, 7m+1:8m)
    L7 .= gram_coeffs[8, 8] .* A6B # low order terms 
    tmpB2 .= gram_coeffs[8, 10] .* A2B + gram_coeffs[8, 12] .* A4B # high order terms bar factor 6 
    mul!(L7, A6, tmpB2, true, true)
    mul!(tmpB2, A, L7, true, false)
    L7 .= tmpB2

    # L9 
    L9 = view(L, 1:n, 9m+1:10m)
    tmpB2 .= gram_coeffs[10, 10] .* A2B + gram_coeffs[10, 12] .* A4B # high order terms bar factor 6 
    mul!(L9, A6, tmpB2, true, false)
    mul!(tmpB2, A, L9, true, false)
    L9 .= tmpB2

    # L11 
    L11 = view(L, 1:n, 11m+1:12m)
    tmpB2 .= gram_coeffs[12, 12] .* A4B # high order terms bar factor 6 
    mul!(L11, A6, tmpB2, true, false)
    mul!(tmpB2, A, L11, true, false)
    L11 .= tmpB2

    # L13 
    L13 = view(L, 1:n, 13m+1:14m)
    tmpB2 .= gram_coeffs[14, 14] .* A6B # high order terms bar factor 6 
    mul!(L13, A6, tmpB2, true, false)
    mul!(tmpB2, A, L13, true, false)
    L13 .= tmpB2

    F = lu!(den)
    expA = num
    ldiv!(F, expA)
    ldiv!(F, L)

    U = qr!(L').R # right Cholesky factor of the Grammian (may not be square!!)
    U = triu2cholesky_factor!(U)
    return expA, U
end


# constructors 

function ExpAndGram{T,3}() where {T}
    pade_num = T[120, 60, 12, 1]
    leg_nums = T[120 0 2 0; 0 60 0 0; 0 0 10 0; 0 0 0 1]
    sqr_norms = T[1, 3, 5, 7]
    normtol = T(0.00067)
    ExpAndGram{T,3,typeof(pade_num),typeof(leg_nums),typeof(sqr_norms)}(
        pade_num,
        leg_nums,
        sqr_norms,
        normtol,
    )
end

function ExpAndGram{T,5}() where {T}
    pade_num = T[30240, 15120, 3360, 420, 30, 1]
    leg_nums = T[
        30240 0 840 0 2 0
        0 15120 0 168 0 0
        0 0 2520 0 10 0
        0 0 0 252 0 0
        0 0 0 0 18 0
        0 0 0 0 0 1
    ]
    sqr_norms = T[1, 3, 5, 7, 9, 11]
    normtol = T(0.021)
    ExpAndGram{T,5,typeof(pade_num),typeof(leg_nums),typeof(sqr_norms)}(
        pade_num,
        leg_nums,
        sqr_norms,
        normtol,
    )
end

function ExpAndGram{T,7}() where {T}
    pade_num = T[17297280, 8648640, 1995840, 277200, 25200, 1512, 56, 1]
    leg_nums = T[
        17297280 0 554400 0 3024 0 2 0
        0 8648640 0 133056 0 324 0 0
        0 0 1441440 0 11880 0 10 0
        0 0 0 144144 0 616 0 0
        0 0 0 0 10296 0 18 0
        0 0 0 0 0 572 0 0
        0 0 0 0 0 0 26 0
        0 0 0 0 0 0 0 1
    ]
    sqr_norms = T[1, 3, 5, 7, 9, 11, 13, 15]
    normtol = T(0.13)
    ExpAndGram{T,7,typeof(pade_num),typeof(leg_nums),typeof(sqr_norms)}(
        pade_num,
        leg_nums,
        sqr_norms,
        normtol,
    )
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
    leg_nums = T[
        17643225600 0 605404800 0 4324320 0 7920 0 2 0
        0 8821612800 0 155675520 0 617760 0 528 0 0
        0 0 1470268800 0 15444000 0 34320 0 10 0
        0 0 0 147026880 0 960960 0 1092 0 0
        0 0 0 0 10501920 0 42120 0 18 0
        0 0 0 0 0 583440 0 1320 0 0
        0 0 0 0 0 0 26520 0 26 0
        0 0 0 0 0 0 0 1020 0 0
        0 0 0 0 0 0 0 0 34 0
        0 0 0 0 0 0 0 0 0 1
    ]
    sqr_norms = T[1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    normtol = T(0.41)
    ExpAndGram{T,9,typeof(pade_num),typeof(leg_nums),typeof(sqr_norms)}(
        pade_num,
        leg_nums,
        sqr_norms,
        normtol,
    )
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
    leg_nums = T[
        64764752532480000 0 2374707592857600 0 21118941043200 0 67044257280 0 81681600 0 32760 0 2 0
        0 32382376266240000 0 647647525324800 0 3620389893120 0 7449361920 0 5569200 0 1080 0 0
        0 0 5397062711040000 0 69390806284800 0 260727667200 0 352716000 0 153000 0 10 0
        0 0 0 539706271104000 0 4797389076480 0 12443820480 0 10852800 0 2380 0 0
        0 0 0 0 38550447936000 0 245321032320 0 439538400 0 232560 0 18 0
        0 0 0 0 0 2141691552000 0 9884730240 0 11938080 0 3344 0 0
        0 0 0 0 0 0 97349616000 0 324498720 0 248976 0 26 0
        0 0 0 0 0 0 0 3744216000 0 8809920 0 3780 0 0
        0 0 0 0 0 0 0 0 124807200 0 197064 0 34 0
        0 0 0 0 0 0 0 0 0 3670800 0 3496 0 0
        0 0 0 0 0 0 0 0 0 0 96600 0 42 0
        0 0 0 0 0 0 0 0 0 0 0 2300 0 0
        0 0 0 0 0 0 0 0 0 0 0 0 50 0
        0 0 0 0 0 0 0 0 0 0 0 0 0 1
    ]
    sqr_norms = T[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27]
    normtol = T(1.57)
    ExpAndGram{T,13,typeof(pade_num),typeof(leg_nums),typeof(sqr_norms)}(
        pade_num,
        leg_nums,
        sqr_norms,
        normtol,
    )
end