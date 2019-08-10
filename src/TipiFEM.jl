#__precompile__(false)

module TipiFEM

if VERSION < v"1.3.0-DEV"
  @warn """TipiFEM with julia version prior to 1.3 is subject to significant
           performance degradation in some situations due to type instability
           issues. Be warned."""
end

# todo: remove
#import Base.eltype
#function eltype(G::Base.Generator)
#    Base.code_typed(G.f,(eltype(G.iter),))[1].rettype
#end

# export submodules
export Meshes, PolytopalMesh
# export types, generic functions
export FEBasis, FESpace, interpolation_nodes, add_constraints!, mark_inactive!,
       constraints, number_of_local_shape_functions, local_shape_functions,
       grad_local_shape_functions, number_of_dofs, matrix_assembler,
       vector_assembler, incorporate_constraints!, l2_norm, dofh,
       interpolation_node_indices, boundary_dofs, BrokenFESpace, DiscontinuousFEFunction,
       project!, sample

include("utils/utils.jl")
include("meshes/meshes.jl")
include("quadrature/quadrature.jl")
#include("simple_1d_mesh/simple_1d_mesh.jl")
include("polytopal_mesh/polytopal_mesh.jl")
include("fe/fe.jl")

# splice local_shape functions into TipiFEM.PolytopalMesh
# todo: a bit too hacky
#let expr = quote
#        using TipiFEM: grad_local_shape_functions, local_shape_functions
#    end
#    Base.eval(TipiFEM.PolytopalMesh, expr)
#end

end
