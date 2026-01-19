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
    gr = (5.0^0.5 - 1.0) / 2.0
    ppi = 72.27


    ncols = 2
    nrows = 1
    w = 5.867722844887229
    h = w * gr * nrows / ncols
    figsize = (ppi * w, ppi * h)

    font = "San Serif"
    fsz = 11
    mainfsz, smallfsz = fsz - 1, fsz - 2
    T = Makie.Theme(
        font = font,
        fontsize = fsz,
        size = figsize,
        figure_padding = 5,
        Axis = (
            titlefont = font,
            subtitlefont = font,
            xlabelfont = font,
            ylabelfont = font,
            xticklabelfont = font,
            yticklabelfont = font,
            titlesize = mainfsz,
            subtitlesize = mainfsz,
            xticklabelsize = smallfsz,
            yticklabelsize = smallfsz,
            xlabelsize = mainfsz,
            ylabelsize = mainfsz,
            spinewidth = 0.5,
            xgridwidth = 0.5,
            ygridwidth = 0.5,
            xtickwidth = 0.5,
            ytickwidth = 0.5,
            xticksize = 2,
            yticksize = 2,
        ),
        Legend = (
            titlefont = font,
            labelfont = font,
            labelsize = mainfsz,
            framewidth = 0.5,
            patchsize = (20, 8),
            padding = (5, 5, 5, 5),
        ),
        Label = (font = font, fontsize = smallfsz),
    )


    nmax = 40
    # setting = standard_setting()
    #T = MakieCore.Theme(setting, ncols = 2, nrows = 1)

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
        Makie.lines!(axleft, results[1], color = "black")
        Makie.lines!(axleft, results[3], color = "black"; linestyle = :dash)
        axright = Axis(
            ga[1, 2],
            yscale = log10,
            xlabel = LaTeXString("\$ n\$"),
            title = LaTeXString("\$ \\text{Cholesky factor} \$"),
        )
        Makie.lines!(axright, results[2], color = "black")
        Makie.lines!(axright, results[4], color = "black"; linestyle = :dash)
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
