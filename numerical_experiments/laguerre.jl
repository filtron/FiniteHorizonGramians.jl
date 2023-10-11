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
    maxn = 100
    λs = [0.5, 1.5, 5.0]
    experiments = [LaguerreExperiment(λ, maxn) for λ in λs]
    results = [run_experiment(experiment) for experiment in experiments]

    setting = standard_setting()
    T = MakieCore.Theme(setting, ncols = length(results), nrows = 1)


    fig = with_theme(T) do
        fig = Figure()
        ga = fig[1, 1] = GridLayout()
        axs = Vector{Axis}(undef, length(λs))
        for (i, result) in pairs(results)
            ax = Axis(ga[1, i], yscale = log10, title = LaTeXString("\$ λ = $(λs[i]) \$"))
            axs[i] = ax
            lines!(ax, eachindex(result), result, color = "black")
            i == 1 || hideydecorations!(ax; grid = false)
        end
        linkyaxes!(axs...)
        colgap!(ga, 10)
        fig
    end
    figname = joinpath(FIG_DIR, "laguerre.pdf")
    Makie.save(figname, fig, pt_per_unit = 1.0)
    display(fig)
end

main()
