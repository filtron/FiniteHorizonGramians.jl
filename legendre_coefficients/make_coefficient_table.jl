import Pkg
Base.active_project() != joinpath(@__DIR__, "Project.toml") && Pkg.activate(@__DIR__)
isfile(joinpath(@__DIR__, "Manifest.toml")) && Pkg.resolve()
Pkg.instantiate()

includet("LegendreCoefficients.jl")
using .LegendreCoefficients, LinearAlgebra, Symbolics, JuliaFormatter


intarray2string(A::AbstractArray{T}) where {T<:Integer} = string(A)
intarray2string(A::AbstractArray{BigInt}) = replace(string(A), "BigInt" => "")

function main()

    filename = joinpath(@__DIR__, "coefficient_table.jl")
    @info "writing coefficient table to $(filename)"
    if isfile(filename)
        rm(filename)
    end

    file = open(filename, "w")
    try
        @variables z
        qs = big.([3, 5, 7, 9, 13])
        for q in qs
            pade_num, leg_nums, sqr_norms = compute_coefficient_table(z, q)

            println(file, "#Legendre expansion of exponential function of order q = $(q)")
            println(file, "pade_num = T" * intarray2string(pade_num))
            println(file, "leg_nums = T" * intarray2string(leg_nums))
            println(file, "sqr_norms = T" * intarray2string(sqr_norms))
            println(file, "")
        end
    finally
        close(file)
        format(filename)
    end

end

!isinteractive() && main()
