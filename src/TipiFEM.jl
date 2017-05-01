#__precompile__()

module TipiFEM

import Base.eltype
eltype(G::Base.Generator) = Base.code_typed(G.f,(eltype(G.iter),))[1].rettype

# export submodules
export Meshes, PolytopalMesh
# export types, generic functions
export FEBasis

include("utils/utils.jl")
include("meshes/meshes.jl")
include("quadrature/quadrature.jl")
include("simple_1d_mesh/simple_1d_mesh.jl")
include("polytopal_mesh/polytopal_mesh.jl")
include("fe/fe.jl")

# splice local_shape functions into TipiFEM.PolytopalMesh
# todo: a bit too hacky
let expr = quote
        using TipiFEM: grad_local_shape_functions, local_shape_functions
    end
    eval(TipiFEM.PolytopalMesh, expr)
end

end
