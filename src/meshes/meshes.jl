module Meshes

export Mesh, add_vertex!, add_cell!, Index, LocalDOFIndex, world_dim, mesh_dim, mesh, index,
       @import_mesh_interface, @export_mesh_interface, skeleton, connectivity,
       cells, Dim, Codim, MeshFunction, add_cell_initializer, vertex,
       Geometry, facet, dim, point, cell_type, real_type, geometry,
       populate_connectivity!, cell_types, complement, facet_count, face_count,
       number_of_cells, topology, reference_element, volume, jacobian_transposed,
       jacobian_inverse_transposed, local_to_global, point, integration_element,
       decompose, vertex_connectivity, elements, nodes, tagged_cells, nodal_coordinates

using StaticArrays
using Iterators
using ComputedFieldTypes

import Base.push!

const cell_initializer = Array{Function, 1}()
function add_cell_initializer(f::Function)
  global cell_initializer
  push!(cell_initializer, f)
end

include("dim.jl")
include("cells/cell.jl")
include("mesh_functions.jl")
include("mesh_topology.jl")
include("mesh.jl")
include("mesh_interface.jl")
include("display.jl")

using TipiFEM.Utils.MethodNotImplemented

# define generic functions to be imported by a mesh implementation
for method in [:reference_element, :facets, :volume, :jacobian_transposed,
               :jacobian_inverse_transposed, :local_to_global,
               :point, :integration_element]
  @eval $(method)(::MethodNotImplemented) = error("Method not implemented")
end

end
