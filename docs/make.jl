using FiniteHorizonGramians
using Documenter

DocMeta.setdocmeta!(FiniteHorizonGramians, :DocTestSetup, :(using FiniteHorizonGramians); recursive=true)

makedocs(;
    modules=[FiniteHorizonGramians],
    authors="Filip Tronarp <filip.tronarp@matstat.lu.se> and contributors",
    repo="https://github.com/filtron/FiniteHorizonGramians.jl/blob/{commit}{path}#{line}",
    sitename="FiniteHorizonGramians.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://filtron.github.io/FiniteHorizonGramians.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/filtron/FiniteHorizonGramians.jl",
    devbranch="main",
)
