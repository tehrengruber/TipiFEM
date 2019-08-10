module PolytopalMesh

using TipiFEM
using TipiFEM.Meshes

export @Polytope_str, @Id_str, @Connectivity_str,
       jacobian_transposed, jacobian_inverse_transposed,
       IntervalMesh, TriangularMesh, HybridMesh2D, global_to_local

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

const IntervalMesh = Mesh{Edge}
const TriangularMesh = Mesh{Triangle}
const HybridMesh2D = Mesh{Union{Triangle, Quadrangle}}

function IntervalMesh(vertex_coordinates::T; kwargs...) where T <: AbstractArray
    msh=IntervalMesh(;kwargs...)
    for x in vertex_coordinates
      add_vertex!(msh, x)
    end
    for (vid1, vid2) in zip(vertices(msh)[1:end-1], vertices(msh)[2:end])
      add_cell!(msh, Polytope"2-node line", vid1, vid2)
    end
    populate_connectivity!(msh)
    msh
end

function IntervalMesh(xl, xr, n; kwargs...)
    vertex_coordinates=Float64[]
    Δx = (xr-xl)/n
    for i in 0:n
        push!(vertex_coordinates, Δx*i)
    end
    vertex_coordinates[end]=xr
    IntervalMesh(vertex_coordinates; kwargs...)
end

end
