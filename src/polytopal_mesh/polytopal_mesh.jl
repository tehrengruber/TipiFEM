module PolytopalMesh

using TipiFEM
using TipiFEM.Meshes

export @Polytope_str, @Id_str, @Connectivity_str, integrate,
       jacobian_transposed, jacobian_inverse_transposed

@import_mesh_interface

abstract type Cell <: Meshes.Cell end

# first define all cell types, their dimension and their topological information
include("cell.jl")
include("dim.jl")
include("conn.jl")

# export mesh interface:
#  exports all cell types
#  generates face_count, facets methods
@export_mesh_interface([Vertex, Edge, Polytope"3-node triangle", Polytope"4-node quadrangle"])

include("geo.jl")
include("topology.jl")
include("quadrature.jl")
include("display.jl")

end
