# activates the script environment and instantiates it
import Pkg
Base.active_project() != joinpath(@__DIR__, "Project.toml") && Pkg.activate(@__DIR__)
haskey(Pkg.project().dependencies, "FiniteHorizonGramians") ||
    Pkg.develop(path = joinpath(@__DIR__, "../"))
isfile(joinpath(@__DIR__, "Manifest.toml")) && Pkg.resolve()
Pkg.instantiate()

using Revise
includet("ExperimentUtilities.jl")
using FiniteHorizonGramians, Random, .ExperimentUtilities
using MakieCore, CairoMakie, TuePlots, LaTeXStrings

function main()

    nmax = 40
    setting = standard_setting()
    T = MakieCore.Theme(setting, ncols = 2, nrows = 1)

    experiment = IntegratorExperiment(nmax)
    results = run_experiment(experiment)

    fig = with_theme(T) do
        fig = Figure()
        ga = fig[1, 1] = GridLayout()
        axleft = Axis(
            ga[1, 1],
            yscale = log10,
            xlabel = LaTeXString("\$ n\$"),
            title = LaTeXString("\$ \\text{Gramian} \$"),
        )
        lines!(axleft, results[1], color = "black")
        lines!(axleft, results[3], color = "black"; linestyle = :dash)
        axright = Axis(
            ga[1, 2],
            yscale = log10,
            xlabel = LaTeXString("\$ n\$"),
            title = LaTeXString("\$ \\text{Cholesky factor} \$"),
        )
        lines!(axright, results[2], color = "black")
        hideydecorations!(axright; grid = false)
        linkyaxes!(axleft, axright)
        colgap!(ga, 10)
        fig
    end


    figname = joinpath(FIG_DIR, "integrator.pdf")
    Makie.save(figname, fig, pt_per_unit = 1.0)
    display(fig)
end

!isinteractive() && main()
