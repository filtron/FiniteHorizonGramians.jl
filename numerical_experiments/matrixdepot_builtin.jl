# activates the script environment and instantiates it 
import Pkg
Base.active_project() != joinpath(@__DIR__, "Project.toml") && Pkg.activate(@__DIR__)
haskey(Pkg.project().dependencies, "FiniteHorizonGramians") ||
    Pkg.develop(path = joinpath(@__DIR__, "../"))
isfile(joinpath(@__DIR__, "Manifest.toml")) && Pkg.resolve()
Pkg.instantiate()

includet("ExperimentUtilities.jl")
using FiniteHorizonGramians, Random, .ExperimentUtilities
using MakieCore, CairoMakie, TuePlots, LaTeXStrings

function main()

    n = 10
    ms = [1, 5, 10]
    nsamples = 50
    α = 1.0 / n

    rng = Random.seed!(19910215)

    setting = standard_setting()
    T = MakieCore.Theme(setting, ncols = length(ms), nrows = 1)


    fig = with_theme(T) do
        fig = Figure()
        ga = fig[1, 1] = GridLayout()
        axs = Vector{Axis}(undef, length(ms))
        for (i, m) in pairs(ms)

            experiment = MatrixDepotBuiltIn(n, m)
            results = [run_experiment(rng, experiment) for _ = 1:nsamples]

            ax = Axis(ga[1, i], yscale = log10, title = LaTeXString("\$ m = $(m) \$"))
            axs[i] = ax
            foreach(results) do result
                scatter!(
                    ax,
                    eachindex(result),
                    result,
                    color = "black",
                    alpha = α,
                    markersize = 3,
                )
                i == 1 || hideydecorations!(ax; grid = false)
            end
        end
        linkyaxes!(axs...)
        colgap!(ga, 10)
        fig
    end
    figname = joinpath(FIG_DIR, "matrixdepot_builtin.pdf")
    Makie.save(figname, fig, pt_per_unit = 1.0)
    display(fig)
end

main()
