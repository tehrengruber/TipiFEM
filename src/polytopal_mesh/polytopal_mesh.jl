module PolytopalMesh

using TipiFEM
using TipiFEM.Meshes

export @Polytope_str, @Id_str, @Connectivity_str, integrate

@import_mesh_interface

abstract type Cell <: Meshes.Cell end

include("cell.jl")
include("dim.jl")
include("geo.jl")
include("conn.jl")
include("topology.jl")
include("quadrature.jl")
include("display.jl")

@export_mesh_interface([Vertex, Edge, Polytope"3-node triangle", Polytope"4-node quadrangle"])

end
