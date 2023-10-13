import Pkg
Base.active_project() != joinpath(@__DIR__, "Project.toml") && Pkg.activate(@__DIR__)
isfile(joinpath(@__DIR__, "Manifest.toml")) && Pkg.resolve()
Pkg.instantiate()

includet("ErrorAnalysis.jl")

using .ErrorAnalysis, MakieCore, CairoMakie, LaTeXStrings, TuePlots

function main()

    # compute maximal norm of 2^(-s)A such that sup_t |V(t, θ)| < 2^(-53)
    T = BigFloat
    qs = big.(1:21)
    order = 150
    ngrid = 2^7
    θs = ErrorAnalysis.backward_bound_exp(T, qs, order)
    ηs = ErrorAnalysis.backward_bound_gram(T, qs, order, ngrid)
    νs = ErrorAnalysis.pade_analytic_radius.(T, qs)
    ξs = map(zip(qs, θs)) do (q, θ)
        ErrorAnalysis.pade_den_bound(T, q, order, θ)
    end

    setting = standard_setting()
    theme = MakieCore.Theme(setting, ncols = 2, nrows = 1)

    fig = with_theme(merge(theme, Theme(Lines = (; cycle = :linestyle)))) do
        fig = Figure()

        ax1 = Axis(
            fig[1, 1],
            #title = LaTeXString("norm bounds"),
            xlabel = LaTeXString("\$q\$"),
            yscale = log2,
        )
        lines!(ax1, Float64.(ηs), label = LaTeXString("\$ \\eta_q \$"))
        lines!(ax1, Float64.(θs), label = LaTeXString("\$ \\theta_q \$"))
        axislegend(ax1, position = :rb)
        ax2 = Axis(fig[1, 2], xlabel = LaTeXString("\$q\$"))
        Δs = @. log2(θs / ηs)
        lines!(ax2, Δs, label = LaTeXString("\$ \\log_2 \\theta - \\log_2\\eta\$"))
        lines!(
            ax2,
            ceil.(Δs),
            label = LaTeXString("\$\\lceil \\log_2 \\theta - \\log_2\\eta\\rceil\$"),
        )
        axislegend(ax2, position = :rt)
        fig
    end

    figname = joinpath(FIG_DIR, "bounds.pdf")
    Makie.save(figname, fig, pt_per_unit = 1.0)
    display(fig)


    filename = joinpath(@__DIR__, "backward_bounds.jl")
    if isfile(filename)
        rm(filename)
    end

    file = open(filename, "w")

    try
        println(file, "θs = T" * string(Float64.(θs)))
        println("")
        println(file, "ηs = T" * string(Float64.(ηs)))
    finally
        close(file)
        format(filename)
    end

end

main()
