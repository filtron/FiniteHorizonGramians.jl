#=
# norm tolls for q in 1:13 for Float62/32
normtols64 = T[
    0.0006794818550677766,
    0.021803366063031033,
    0.13306928209763041,
    0.41098379928173995,
    1.579470165477942,
]

normtols32 = T[
    0.048,
    0.61190283,
    1.5580196,
    2.801222,
    5.7639575,
]
=#

_normtol(::Type{T}, ::Val{3}) where {T<:Float32} = 0.048
_normtol(::Type{T}, ::Val{5}) where {T<:Float32} = 0.61190283
_normtol(::Type{T}, ::Val{7}) where {T<:Float32} = 1.5580196
_normtol(::Type{T}, ::Val{9}) where {T<:Float32} = 2.801222
_normtol(::Type{T}, ::Val{13}) where {T<:Float32} = 5.7639575

_normtol(::Type{T}, ::Val{3}) where {T<:Float64} = 0.0006794818550677766
_normtol(::Type{T}, ::Val{5}) where {T<:Float64} = 0.021803366063031033
_normtol(::Type{T}, ::Val{7}) where {T<:Float64} = 0.13306928209763041
_normtol(::Type{T}, ::Val{9}) where {T<:Float64} = 0.41098379928173995
_normtol(::Type{T}, ::Val{13}) where {T<:Float64} = 1.579470165477942

# this is a bit hacky but should work for Floats / ComplexFloats / Duals 
_normtol(::Type{T}, ::Val{Q}) where {T,Q} = _normtol(typeof(eps(real(float(T)))), Val(Q))


