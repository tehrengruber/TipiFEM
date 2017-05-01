using Documenter, TipiFEM

makedocs(
    modules = [TipiFEM, TipiFEM.Meshes],
    format = :html,
    sitename = "TipiFEM.jl",
    pages = Any[
        "Home" => "index.md"
    ],
    clean = false,
    #doctest = false
)
