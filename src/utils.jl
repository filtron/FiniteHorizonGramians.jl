"""
    _dims_if_compatible(A::AbstractMatrix, B::AbstractMatrix)

Throws DimensionMismatch if A is not square or if the number of rows of B does not equal 
the number of columns of A. 
Is equivalent to size(B) if no error is thrown. 
"""
function _dims_if_compatible(A::AbstractMatrix, B::AbstractMatrix)
    n, m = size(B)
    n == LinearAlgebra.checksquare(A) || throw(
        DimensionMismatch(
            "size of A, $(LinearAlgebra.checksquare(A)), incompatible with the size of B, $(size(B)).",
        ),
    )
    return n, m
end


"""
    triu2cholesky_factor!(A::AbstractMatrix{T})

If A is an upper triangular matrix, it computes the product Q*A in-place, 
where Q is a unitary transform such that Q*A is a valid Cholesky factor. 
If A is not an upper triangular matrix, returns garbage. 
"""
function triu2cholesky_factor!(A::AbstractMatrix{T}) where {T<:Number}
    LinearAlgebra.require_one_based_indexing(A)
    nrow, ncol = size(A)
    for row = 1:nrow
        conjsign = sign(A[row, row])'
        for col = row:ncol
            A[row, col] = conjsign * A[row, col]
        end
    end
    return A
end


"""
    _symmetrize!(A::AbstractMatrix{T}) where {T<:Number}

Discards the skew-Hermitian part of A in-place. 
"""
function _symmetrize!(A::AbstractMatrix{T}) where {T<:Number}
    LinearAlgebra.require_one_based_indexing(A)
    n = LinearAlgebra.checksquare(A)
    for col = 1:n
        for row = 1:col
            A[row, col] = (A[row, col] + A[col, row]') / 2
            if row != col
                A[col, row] = A[row, col]
            end
        end
    end
end
