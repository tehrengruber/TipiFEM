module Meshes

export Mesh, add_vertex!, add_cell!, Id, LocalDOFIndex, world_dim, mesh_dim, mesh, index,
       @import_mesh_interface, @export_mesh_interface, skeleton, connectivity,
       cells, Dim, Codim, MeshFunction, add_cell_initializer, vertex,
       Geometry, facet, dim, point, cell_type, real_type, geometry,
       populate_connectivity!, cell_types, complement, facet_count, face_count,
       number_of_cells, topology, reference_element, volume, jacobian_transposed,
       jacobian_inverse_transposed, local_to_global, point, integration_element,
       decompose, vertex_connectivity, elements, nodes, tagged_cells, vertex_coordinates,
       IdIterator, HomogeneousIdIterator,HeterogenousIdIterator, GenericIdIterator,
       issimple, parent_type, hasparent, world_dim, cell_type,
       set_domain!, set_image!, tag_cells!, tag_vertices!, number_of_elements, Cell,
       is_cannonical, canonicalize_connectivity, flip_orientation, element_type, normal

using StaticArrays
using ComputedFieldTypes

import Base: push!

const cell_initializer = Array{Tuple{Function, Dict{Symbol, Any}}, 1}()
function add_cell_initializer(f::Function; kwargs...)
  global cell_initializer
  kwargs=merge(Dict{Symbol, Any}(:hybrid=>false), kwargs)
  push!(cell_initializer, (f, kwargs))
end

include("dim.jl")
include("cells/cell.jl")
include("mesh_functions.jl")
include("mesh_topology.jl")
include("mesh_datastructure.jl")
include("mesh_interface.jl")
include("display.jl")
include("one_time_cell_visitor.jl")

using TipiFEM.Utils: MethodNotImplemented

const registered_cell_types = Array{Type{<:Cell}, 1}()

# define generic functions to be imported by a mesh implementation
for method in [:reference_element, :facets, :volume, :jacobian_transposed,
               :jacobian_inverse_transposed, :local_to_global,
               :point, :integration_element, :is_cannonical,
               :canonicalize_connectivity, :flip_orientation, :normal]
  @eval $(method)(::MethodNotImplemented) = error("Method not implemented")
end

end
