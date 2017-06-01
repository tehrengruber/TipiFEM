using StaticArrays
export facets

using TipiFEM.Utils: @generate_sisd

"Geometry of the reference triangle"
function reference_element(::Polytope"3-node triangle")
  Geometry{Polytope"3-node triangle", 2, Float64}(
    SVector{2, Float64}(0, 0),
    SVector{2, Float64}(1, 0),
    SVector{2, Float64}(0, 1))
end

"Geometry of the reference quadrangle"
function reference_element(::Polytope"4-node quadrangle")
  Geometry{Polytope"4-node quadrangle", 2, Float64}(
    SVector{2, Float64}(0, 0),
    SVector{2, Float64}(1, 0),
    SVector{2, Float64}(1, 1),
    SVector{2, Float64}(0, 1))
end

# todo: rewrite geometry and topology reduction by this simple rules
#facets(::Polytope"3-node triangle") = (Edge, (1, 2), (2, 3), (3, 1))
#
#facets(::Polytope"4-node quadrangle") = (Edge, (1, 2), (2, 3), (3, 4), (4, 1))

"Geometry of the facets of a triangle"
function facets{G <: Geometry{Polytope"3-node triangle"}}(geo::G)
  (Geometry{Edge, world_dim(G), real_type(G)}(point(geo, 1), point(geo, 2)),
   Geometry{Edge, world_dim(G), real_type(G)}(point(geo, 2), point(geo, 3)),
   Geometry{Edge, world_dim(G), real_type(G)}(point(geo, 3), point(geo, 1)))
end

"Geometry of the facets of a quadrangle"
function facets{G <: Geometry{Polytope"4-node quadrangle"}}(geo::G)
  (Geometry{Edge, world_dim(G), real_type(G)}(point(geo, 1), point(geo, 2)),
   Geometry{Edge, world_dim(G), real_type(G)}(point(geo, 2), point(geo, 3)),
   Geometry{Edge, world_dim(G), real_type(G)}(point(geo, 3), point(geo, 4)),
   Geometry{Edge, world_dim(G), real_type(G)}(point(geo, 4), point(geo, 1)))
end

function find_longest_edge(geo::G) where G <: Geometry{Polytope"3-node triangle"}
  edges = facets(geo)
  edge_lengths = map(volume, edges)
  idx = indmax(edge_lengths)
  egeo = edges[idx]
  (idx, egeo)
end

function find_longest_edge(conn::C, geo::G) where {C <: Connectivity"3-node triangle → 1-node point", G <: Geometry{Polytope"3-node triangle"}}
  idx, egeo = find_longest_edge(geo)
  (idx, facets(conn)[idx], egeo)
end

# todo: local indices differ to refine(conn, geo)
function refine{G <: Geometry{Polytope"3-node triangle"}}(geo::G)
  const edge_t = Geometry{Polytope"2-node line", world_dim(G), real_type(G)}
  edges = facets(ref_tria)
  # find longest edge
  longest_edge_index, longest_edge = find_longest_edge(geo)
  # split lonest edge
  longest_edge_midpoint = (point(longest_edge, 1)+point(longest_edge, 2))/2
  # determine vertex opposite to the longest edge
  opposite_vertex = point(geo, (3+longest_edge_index-1)%3)
  # refined triangles
  triangle_1 = G(opposite_vertex,
                 point(longest_edge, 1),
                 longest_edge_midpoint)
  triangle_2 = G(opposite_vertex,
                 longest_edge_midpoint,
                 point(longest_edge, 2))
  (triangle_1, triangle_2)
end

function refine!(mesh::Mesh, id::Index{C}) where C <: Cell
  mesh_conn = vertex_connectivity(mesh, C())
  conn = mesh_conn[id]
  geo = geometry(mesh, id)
  # find longest edge
  longest_edge_index, longest_edge, longe_edge_geo = find_longest_edge(conn, geo)
  # add new midpoint of the longest edge to the mesh
  push!(mesh.nodes, (point(longe_edge_geo, 1)+point(longe_edge_geo, 2))/2)
  longest_edge_midpoint = last(domain(mesh.nodes))
  # deduce vertex opposite to the longest edge
  opposite_vertex = conn[(3+longest_edge_index-1)%3]

  # construct connectivity of the refined triangles
  triangle_1 = Connectivity{C, Vertex}(opposite_vertex, longest_edge[1], longest_edge_midpoint)
  triangle_2 = Connectivity{C, Vertex}(opposite_vertex, longest_edge_midpoint, longest_edge[2])
  mesh_conn[id] = triangle_1
  add_cell!(mesh, triangle_2)
end

"Length of a line"
volume(geo::Geometry{Polytope"2-node line"}) = norm(point(geo, 1) - point(geo, 2))

"Area of a triangle"
@inline volume{G <: Geometry{Polytope"3-node triangle"}}(geo::G) = abs(det(SMatrix{3, 3, real_type(G)}(
  one(real_type(G)), point(geo, 1)[1], point(geo, 1)[2],
  one(real_type(G)), point(geo, 2)[1], point(geo, 2)[2],
  one(real_type(G)), point(geo, 3)[1], point(geo, 3)[2],
)))/2

"Area of a quadrangle"
function volume(geo::Geometry{Polytope"4-node quadrangle"})
  # todo: implement
  error("not implemented")
end

"""
Integration element of the geometry mapping

In other words the determinant of the jacobian of of the geometry mapping.
"""
function integration_element{n}(geo::Geometry{Polytope"3-node triangle"},
    x::SMatrix{n, 2, <:Real})
  volume(geo)*2*ones(SVector{n, eltype(x)})
end

integration_element(geo::Geometry{Polytope"3-node triangle"}) = volume(geo)*2

function integration_element(geo::Geometry{Polytope"3-node triangle"}, x::SVector{2,<:Real})
  volume(geo)*2
end

function integration_element{n}(geo::Geometry{Polytope"4-node quadrangle"},
    x::SMatrix{n, 2, <:Real})
  det(jacobian_transposed(geo, x[1, :]))*ones(SVector{n, eltype(x)})
end

function integration_element(geo::Geometry{Polytope"4-node quadrangle"}, x::SVector{2,<:Real})
  det(jacobian_transposed(geo, x))
end

"""
Maps local to global coordinates or in other words given a set of coordinates
on the reference element map them to coordinates on `G`.

Φ: K̂ → K with K̂ ⊂ ℝ^local_dim, K ⊂ ℝ^world_dim
"""
@generate_sisd function local_to_global(geo::Geometry{C, world_dim},
    x̂s::SMatrix{@VectorWidth(n), local_dim, T}) where {C <: Polytope, world_dim, local_dim, n, T<:Real}
  # old version: local_shape_functions(FEBasis{:Lagrangian, 1}(), C(), x̂s)*geo
  # todo: benchmark in comparision to the old version

  # allocate result
  # todo:
  global_coords = zeros(MMatrix{n, world_dim})
  # evaluate local shape functions
  #  the i-th index containts the values of the i-th local shape function
  lsfns = local_shape_functions(FEBasis{:Lagrangian, 1}(), C(), x̂s)
  for d in 1:world_dim
    for i in 1:vertex_count(geo)
      global_coords[:, d] += geo[i, d] * lsfns[i]
    end
  end
  SMatrix{n, world_dim}(global_coords)
end

#function local_to_global{G <: Geometry{Polytope"3-node triangle"}, REAL_ <: Real}(geo::G, x::SVector{2, REAL_})
#  F_K = hcat(point(geo, 2)-point(geo, 1), point(geo, 3)-point(geo, 1))
#  τ_K = point(geo, 1)
#  F_K*x+τ_K
#end

# dimFrom -> cell_dim / local_dim
# dimTo -> world_dim

function jacobian_transposed(geo::Geometry{C, world_dim},
    x̂::SVector{local_dim, T}) where {C <: Polytope, world_dim, local_dim, T<:Real}
  #jacobian = MMatrix{world_dim, local_dim, eltype(x̂)}
  grads = grad_local_shape_functions(FEBasis{:Lagrangian, 1}(), C(), x̂)
  grads_mat = hcat(grads...)

  res = grads_mat*geo
  res
end

function jacobian_inverse_transposed(geo::Geometry{C, world_dim},
    x̂::SVector{local_dim, T}) where {C <: Polytope, world_dim, local_dim, T<:Real}
  inv(jacobian_transposed(geo, x̂))
end

#@generate_sisd function jacobian_transposed(geo::Geometry{C, world_dim},
#    x̂s::SMatrix{@VectorWidth(n), local_dim, T}) where {C <: Cell, world_dim, local_dim, n, T<:Real}
#  # datatype of the jacobian
#  #jacobian_t = SMatrix{world_dim, local_dim, eltype(x̂s)}
#  # initialize 3 dimensional array containing the jacobians
#  JTs = MArray{local_dim, num_points, world_dim}()
#  # evaluate the gradients at x̂s
#  grads = grad_local_shape_functions(FEBasis{:Lagrangian, 1}(), C(), x̂s)
#  # calculate jacobians
#  for i in 1:local_dim
#    for j in 1:world_dim
#      JTs[:, i, j] = reduce(+, grads.*geo[:, j])
#    end
#  end
#end

#function local_to_global{G <: Geometry{Polytope"3-node triangle"}}(geo::G, x::SVector{2, <:Real})
#  F_K = hcat(point(geo, 2)-point(geo, 1), point(geo, 3)-point(geo, 1))
#  τ_K = point(geo, 1)
#  F_K*x+τ_K
#end

midpoint(geo::Geometry{Polytope"2-node line"}) = (point(geo, 1)+point(geo, 2))/2

function midpoint_quadrature_rule{G <: Geometry{Polytope"3-node triangle"}}(geo::G, f::Function)
  sum(map(e -> f(midpoint(e)), facets(geo)))/6
end

function local_midpoint_quad_rule{G <: Geometry{Polytope"3-node triangle"}}(::Type{G}, f::Function)
  ref_geo = reference_element(G)
  sum(map(e -> f(midpoint(e)), facets(ref_geo)))/6
end
