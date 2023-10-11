using LinearAlgebra, BenchmarkTools



function schur_reduce(ΠU::F, C::AbstractMatrix{T}, RU::F) where {T,F<:UpperTriangular{T}}
    m, n = size(C)

    # form pre array 
    pre_array = similar(C, m + n, m + n)
    pre_array[1:m, 1:m] .= RU' 
    pre_array[m+1:m+n, 1:m] .= ΠU'* C'
    pre_array[1:m, m+1:m+n] .= zero(T)
    pre_array[m+1:m+n, m+1:m+n] .= ΠU'
    
    # form post array and fix sign 
    post_array = qr(pre_array).R 
    for row = 1:m+n
        conjsign = sign(post_array[row, row])'
        for col = row:m+n
            post_array[row, col] = conjsign * post_array[row, col]
        end
    end

    # form output 
    SU = UpperTriangular(post_array[1:m, 1:m]) # copies 
    ΣU = UpperTriangular(post_array[m+1:m+n, m+1:m+n]) # copies 
    Kadj = post_array[1:m, m+1:m+n] # copies 
    K = Kadj'
    #K = K / SU'
    K = rdiv!(K, SU')
    return SU, K, ΣU  
end

function schur_reduce!(SU::F, K::AbstractMatrix{T}, ΣU::F, ΠU::F, C::AbstractMatrix{T}, RU::F, pre_array) where {T,F<:UpperTriangular{T}}
    m, n = size(C)

    # form pre array 
    @inbounds pre_array[1:m, 1:m] .= RU' 
    @inbounds mul!(view(pre_array, m+1:m+n, 1:m), ΠU', C')
    @inbounds pre_array[1:m, m+1:m+n] .= zero(T)
    @inbounds pre_array[m+1:m+n, m+1:m+n] .= ΠU'
    
    # form post array and fix sign 
    post_array = qr!(pre_array).R 
    @inbounds for row = 1:m+n
        conjsign = sign(post_array[row, row])'
        @inbounds for col = row:m+n
            post_array[row, col] = conjsign * post_array[row, col]
        end
    end

    # form output 
    @inbounds SU.data .= post_array[1:m, 1:m] 
    @inbounds ΣU.data .= post_array[m+1:m+n, m+1:m+n]  
    @inbounds Kadj = view(post_array, 1:m, m+1:m+n) 
    K .= Kadj'
    rdiv!(K, SU')
    return SU, K, ΣU  
end



function main()
    n = 100 
    m = 100 
    ΠU = triu(ones(n, n)) |> UpperTriangular
    RU = triu(ones(m, m)) |> UpperTriangular
    C = randn(m, n) 

    SU = similar(RU)
    K = similar(C, n, m)
    ΣU = similar(ΠU)
    pre_array = similar(C, m + n, m + n)
    @btime schur_reduce($ΠU, $C, $RU)
    @btime schur_reduce!($SU, $K, $ΣU, $ΠU, $C, $RU, $pre_array)

end
