module PolytopalMesh

using TipiFEM
using TipiFEM.Meshes

export @Polytope_str, @Id_str, @Connectivity_str,
       jacobian_transposed, jacobian_inverse_transposed

@import_mesh_interface

abstract type Cell <: Meshes.Cell end

# first define all cell types, their dimension and topological information
include("cell.jl")
include("dim.jl")
include("conn.jl")

# export mesh interface:
#  exports all cell types
#  generates face_count, facets methods
@export_mesh_interface([Vertex, Edge, Triangle, Quadrangle, Union{Triangle, Quadrangle}])

include("geo.jl")
include("topology.jl")
include("quadrature.jl")
include("display.jl")

end
