module Simple1DMesh

using TipiFEM
using TipiFEM.Meshes
using StaticArrays

@import_mesh_interface

abstract type Cell <: Meshes.Cell end

# define what a vertex is
struct Vertex <: Cell end
dim(::Type{Vertex}) = 0

# define what an edge is
struct Edge <: Cell end
dim(::Type{Edge}) = 1
face_count(::Type{Edge}, ::Type{Vertex}) = 2
facet(::Type{Edge}) = Vertex
volume(geo::Geometry{Edge}) = point(geo, 2)[1] - point(geo, 1)[1]

# define what a triangle is
#struct Triangle <: Cell end
#dim(::Type{Triangle}) = 2
#vertex_count(::Type{Triangle}) = 3
#facet(::Type{Triangle}) = Edge
#facets(tri::Connectivity{Triangle, Vertex}) = (Connectivity{Edge, Vertex}(vertex(tri, 1), vertex(tri, 2)),
#                                               Connectivity{Edge, Vertex}(vertex(tri, 2), vertex(tri, 3)),
#                                               Connectivity{Edge, Vertex}(vertex(tri, 3), vertex(tri, 1)))
#reference_element(::Type{Triangle}, ::Type{Connectivity}) = Connectivity{Triangle, Vertex}(1, 2, 3)
#
#function reference_element(::Type{Triangle}, ::Type{Geometry}; real_type::DataType=Float64)
#  Geometry{Triangle, Float64}(SVector{3, real_type}(0, 0),
#                              SVector{3, real_type}(1, 0),
#                              SVector{3, real_type}(0, 1))
#end
#
#volume(geo::Geometry{Triangle}) = det(SMatrix{3, 3}(
#  1, point(geo, 1)...,
#  1, point(geo, 2)...,
#  1, point(geo, 3)...
#)

#facets(mesh, idx::Id{Triangle}) = connectivity(mesh, Triangle, facet(Triangle))[idx]
#cofacets(mesh, idx::CanonicalId{Edge}) = connectivity(mesh, Edge, cofacet(Edge))[idx]
#cofacet(mesh, idx::Id{Edge}) = connectivity(mesh, Edge, cofacet(Edge))[idx]


#boundary(t::Triangle) = (Connectivity{Edge, Vertex}(vertex(tri, 1), vertex(tri, 2)),
#                         Connectivity{Edge, Vertex}(vertex(tri, 2), vertex(tri, 3)),
#                         Connectivity{Edge, Vertex}(vertex(tri, 3), vertex(tri, 1)))

#struct FEBasis{K <: Cell, typ, degree}
#end

#dof(conn::Connectivity{Edge, Vertex}, ::FEBasis{Edge, :Lagrangian, :Linear}, local_index) =
#  convert(Int, conn[local_index])
#
#@with_mesh(mesh) do
#  for cell in cells(Codim{0})
#    vertices(cell)
#    index(cell)
#    geometry(cell)
#  end
#
#  foreach(cells(mesh, Codim{0})) do cell
#    cell
#  end
#end
#
#for cell in cells(mesh)
#  vertices(cell)
#  index(cell)
#  geometry(cell)
#
#cells(mesh, (:connectivity, :geometry, (e) -> volume(e)))
#
#local_shape_function(::FEBasis{Edge, :Lagrangian, :Linear}, ::LocalDOFId{1}) = x -> 2(1-x)-1
#local_shape_function(::FEBasis{Edge, :Lagrangian, :Linear}, ::LocalDOFId{2}) = x -> 2x

@export_mesh_interface(Cell)

end
