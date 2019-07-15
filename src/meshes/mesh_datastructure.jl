using StaticArrays
using SimpleRepeatIterator
using TipiFEM.Utils: type_scatter, flatten
export subcell, world_dim, number_of_vertices

# todo: add invalid flag to all cell entities

using Base: OneTo

mutable struct MeshCacheEntry
  valid::Bool
  data
end

"""

```
  Mesh(::Type{K}; simple=Val{true}, world_dim=dim(K))
```

Initialize a mesh containing `K` cells. Note that `K` may also be a union of
(compatible) cell types.

# Definitions

We say that a mesh is `simple` if the the mesh function storing the vertex
coordinates is simple, i.e. its domain is a UnitRange object. This property
enables adding vertices to the mesh without explicitly allocating or specifying
vertex ids.

# Examples

__Mesh initialization with different element types__
```
Mesh(Polytope"2-node line") # 1D
Mesh(Polytope"3-node triangle") # 2D
Mesh(Union{Polytope"3-node triangle", Polytope"4-node quadrangle"}) # 2D hybrid
```

__Hybrid mesh construction__
```
# construct mesh instance
julia> mesh = Mesh(Union{Polytope"3-node triangle",Polytope"4-node quadrangle"});
TipiFEM.Mesh
  world dim: 2
  mesh dimension: 2
  cell types: Polytope"3-node triangle", Polytope"4-node quadrangle"
   Codim    Dim     #cells
       0      2          0
       1      1          0
       2      0          0
  topology:
    → | 0  1  2
    --|--------
    0 | 0  0  0
    1 | 0  0  0
    2 | 1  0  0

# add some vertices
julia> add_vertex!(mesh, 0, 0);
julia> add_vertex!(mesh, 0, 1);
julia> add_vertex!(mesh, 1, 1);
julia> add_vertex!(mesh, 1, 0);
julia> add_vertex!(mesh, 3, 0.5)
5 element HomogenousMeshFunction Polytope"1-node point" → StaticArrays.SArray{Tuple{2},Float64,1,2}
 (1, [0.0, 0.0])
 (2, [0.0, 1.0])
 (3, [1.0, 1.0])
 (4, [1.0, 0.0])
 (5, [3.0, 0.5])

# query the number of cells
julia> number_of_cells(mesh, Polytope"1-node point")
5

# connect vertices
julia> add_cell!(mesh, Polytope"4-node quadrangle", 1, 2, 3, 4);
julia> add_cell!(mesh, Polytope"3-node triangle", 2, 5, 3)
2 element HeterogenousMeshFunction Union{Polytope"3-node triangle", Polytope"4-node quadrangle"} → Union{Connectivity{Polytope"3-node triangle",Polytope"1-node point",3,3}, Connectivity{Polytope"4-node quadrangle",Polytope"1-node point",4,4}}
├─ 1 element HomogenousMeshFunction Polytope"3-node triangle" → Connectivity{Polytope"3-node triangle",Polytope"1-node point",3,3}
|   (1, Id{Polytope"1-node point"}[2, 5, 3])
├─ 1 element HomogenousMeshFunction Polytope"4-node quadrangle" → Connectivity{Polytope"4-node quadrangle",Polytope"1-node point",4,4}
|   (1, Id{Polytope"1-node point"}[1, 2, 3, 4])

# populate topology (restricted bidirectional by default)
julia> populate_connectivity!(mesh);
```

"""
@computed struct Mesh{K, world_dim, REAL_ <: Real, simple, PM}
  nodes::HomogenousMeshFunction{vertex(K), world_dim==1 ? REAL_ : SVector{world_dim, REAL_},
    simple ? OneTo{Id{vertex(K)}} : Vector{Id{vertex(K)}}, Vector{world_dim==1 ? REAL_ : SVector{world_dim, REAL_}}}
  topology::fulltype(MeshTopology{K, simple})
  attributes::Dict{Any, MeshFunction}
  cell_groups::Dict{Any, Vector} # todo: use Vector{<:Id}
  cache::Dict{Symbol, MeshCacheEntry}
  parent::PM

  function Mesh(parent::PM=nothing)
    mesh = new{K, world_dim, REAL_, simple, PM}(
      MeshFunction(vertex(K), world_dim==1 ? REAL_ : SVector{world_dim, REAL_}, Val{simple}()),
      MeshTopology(K, Val{simple}()),
      Dict{Symbol, MeshFunction}(),
      Dict{Any, Array{Cell, 1}}(),
      Dict{Symbol, MeshCacheEntry}(),
      #Dict{Symbol, Any}(),
      parent)
    mark_populated!(mesh.topology, dim_t(K), Dim{0}())
    #attributes(msh)[:proper] = true
    mesh
  end
end

@generated function (::Type{Mesh{K}})(;simple=Val{true}(), world_dim=nothing) where K <: Cell
  # parse world dimension argument
  if world_dim == Nothing
    world_dim_expr = :(dim(K))
  elseif world_dim <: Val && isa(tparam(world_dim, 1), Number)
    world_dim_expr = tparam(world_dim, 1)
  else
    throw("Type of `world_dim` argument must be of type Val{N} with N <: Integer")
  end

  :(Mesh{K, $(world_dim_expr), Float64, simple==Val{true}(), Nothing}())
end

mesh_cache_initializer = Dict{Symbol, Function}()

function add_cache_initializer(f::Function, s::Symbol)
  mesh_cache_initializer[s] = f
end

function cache_entry(msh::Mesh, s::Symbol)
  if !haskey(msh.cache, s) || !msh.cache[s].valid
    data = mesh_cache_initializer[s](msh)
    msh.cache[s] = MeshCacheEntry(false, data)
    return data
  end

  msh.cache[s].data
end

# Default cache initializers
#
# :unique_cell_visitor cache - mesh function from all cells to symbols
#  usually used in loops over a set of cells whose subcells shall only be
#  processed once. The idea is to generate a unique symbol in the beginning
#
# mesh.cache[:unique_cell_visitor]
# for cid in connectivity(mesh, some_cells)
#   for skeleton(cid)
#
#
#

function subcells(mesh::Mesh, cells::IdIterator)
  let subcells = HeterogenousVector{eltype(cells)}()
    visitor_id = mesh.cache[:unique_cell_visitor].visitor_id
    foreach(cells) do cid
      if mesh.cache[:unique_cell_visitor][cid] != visitor_id
        push!(subcells[cell_type(cid)], cid)
        mesh.cache[:unique_cell_visitor][cid] = visitor_id
      end
    end
    subcells
  end
end

# ===============================================================================
# Mesh metadata accessors
# ===============================================================================

"""
    real_type(::Union{Mesh, Type{<:Mesh}})

Return type used for calculations with real numbers.
"""
@typeinfo real_type(M::Type{<:Mesh}) = tparam(M, 3)

"""
    issimple(mesh::Union{Mesh, Type{<:Mesh}})

Determine whether the `mesh` is simple.
"""
@typeinfo issimple(M::Type{<:Mesh}) =  tparam(M, 4)

"""
    parent_type(mesh::Union{Mesh, Type{<:Mesh}})

Return the type of the parent mesh.
"""
@typeinfo parent_type(M::Type{<:Mesh}) = tparam(M, 5)

"""
    hasparent(mesh::Union{Mesh, Type{<:Mesh}})

Determine whether the `mesh` has a parent.
"""
@typeinfo hasparent(M::Type{<:Mesh}) = parent_type(M) != Void

"""
    world_dim(mesh::Union{Mesh, Type{<:Mesh}})

Dimension of the ambient space

```
# create a mesh with world dimension 3 and topological dimension 1
julia> mesh=Mesh(Polytope"2-node line", world_dim=Val{3}())
julia> world_dim(mesh)
3
```
"""
@typeinfo world_dim(M::Type{<:Mesh}) = tparam(M, 2)

"""
    mesh_dim(M::Union{Mesh, Type{<:Mesh}})

dimension of codimension zero cells
"""
@typeinfo mesh_dim(M::Type{<:Mesh}) = dim(element_type(M))

"""
    cell_type(mesh::Union{Mesh, Type{<:Mesh}}, d)

Type of cells with (co)-dimension `d`.
"""
@typeinfo cell_type(M::Type{<:Mesh}, d) = let K = tparam(M, 1)
  subcell(K, d)
end

"""
    element_type(mesh::Union{Mesh, Type{<:Mesh}})

Element type, i.e. type of codim zero cells

```
julia> element_type(Mesh(Polytope"3-node triangle"))
Polytope"3-node triangle"
```
"""
@typeinfo element_type(M::Type{<:Mesh}) = let K = tparam(M, 1)
  K
end

"""
    cell_types(M::Type{<:Mesh}, d=Codim{0}())

Type of cells in `M` as a tuple, with `d` the (co)-dimension

```
julia> cell_types(Mesh(Union{Polytope"3-node triangle", Polytope"4-node quadrangle"}))
(Polytope"3-node triangle", Polytope"4-node quadrangle")
```
"""
@typeinfo cell_types(M::Type{<:Mesh}, d=Codim{0}()) = (Base.uniontypes(cell_type(M, d))...,)

# ===============================================================================
# Mesh field accessors
# ===============================================================================
import Base: parent

parent(mesh::Mesh) = mesh.parent

topology(mesh) = mesh.topology


# ===============================================================================
# Mesh information
# ===============================================================================

"""
    number_of_cells(msh::Mesh, d::Union{Dim, Codim})

Number of `d` dimensional cells in `msh`
"""
@dim_dispatch function number_of_cells(msh::Mesh{K}, d::Codim) where K <: Cell
  # if the number of vertices is requested we have to look at the nodes arrays
  if complement(element_type(msh), d)==Dim{0}()
    length(msh.nodes)
  # otherwise the number of cells is stored in the topology array
  else
    ispopulated(topology(msh), d, Dim{0}()) ? length(connectivity(msh, d, Dim{0}())) : 0
  end
end

"""
    number_of_vertices(msh::Mesh)

Number of vertices in `msh`
"""
number_of_vertices(msh::Mesh) = number_of_cells(msh, Dim{0}())

"""
    number_of_elements(msh::Mesh)

Number of codim zero cells in `msh`
"""
number_of_elements(msh::Mesh) = number_of_cells(msh, Codim{0}())

"""
    number_of_cells(msh::Mesh, ::Union{T, Type{T}})

Number of cells of type `T` in `msh`
"""
function number_of_cells(msh::Mesh, ::Union{T, Type{T}}) where T <: Cell
  if dim(T)==0
    # return the number of vertices
    length(msh.nodes)
  else
    # check that the entry in the topology array is initialized
    ispopulated(topology(msh), dim_t(T), Dim{0}()) ||
      error("number of cells is not known yet. Did you forget to call populate_connectivity!?")
    # return the number of cells
    length(connectivity(msh, dim_t(T), Dim{0}())[T])
  end
end

# ===============================================================================
# Mesh topology
# ===============================================================================

#function set_connectivity!(mesh, d1::Int, d2::Int, mf::MeshFunction)
#  # todo: check type of the mesh function
#  mesh.topology[Dim{d1}(), Dim{d2}()] = mf
#end

"""
    connectivity(mesh::Mesh{K}, d1::Dim{i}, d2::Dim{j}) where {K <: Cell, i, j}

Return the connectivity of `i` dimensional cells with `j` dimensional cells, i.e.
a representation of the incidence relation `i` → `j`
"""
@dim_dispatch function connectivity(mesh::Mesh{K}, d1::Dim{i}, d2::Dim{j}) where {K <: Cell, i, j}
  topology(mesh)[d1, d2]
end

"""
    connectivity(mesh::Mesh{K}, ::C, j::Dim) where {K <: Cell, C <: Cell}

Return the connectivity of cells of type `C` with `j` dimensional cells.
"""
@dim_dispatch function connectivity(mesh::Mesh{K}, ::C, j::Dim) where {K <: Cell, C <: Cell}
  #assert(subcell(C1, dim_t(C2)) == C2)
  connectivity(mesh, dim_t(C), j)[C]
end

"""
    connectivity(mesh::Mesh{K}, ::C, j::Dim) where {K <: Cell, C <: Cell}

Return the connectivity of cells of type `C` with vertices, i.e. for each `C`
cell its vertices.
"""
function vertex_connectivity(mesh::Mesh, ::C) where C <: Cell
  connectivity(mesh, C(), subcell(C, Dim{0}())())
end

"""
    vertex_connectivity(mesh::Mesh)

Return the connectivity of all elements with their vertices, i.e. for each
element its vertices.
"""
vertex_connectivity(mesh::Mesh) = connectivity(mesh, Codim{0}(), Dim{0}())

"""
    connectivity(mesh::M, ids::IdIterator{C}) where {C <: Cell, M <: Mesh}

Return the connectivity of all cells contained in `ids` with their vertices.
"""
function connectivity(mesh::M, ids::HomogeneousIdIterator{C}) where {C <: Cell, M <: Mesh}
  @assert typeof(C) == DataType "Only a single cell type allowed"
  let vertex_coordinates = vertex_coordinates(mesh),
      mesh_conn = connectivity(mesh, C(), Dim{0}())
    mesh_geo = MeshFunction(C, Geometry{C, world_dim(M), real_type(M)}, Val{false}())
    #resize!(mesh_geo, length(ids))
    for id in ids
      push!(mesh_geo, id, map(vidx -> vertex_coordinates[vidx], mesh_conn[id]))
    end
    mesh_geo
  end
end

function connectivity(mesh::Mesh, ids::HeterogenousIdIterator)
  compose(map(ids -> geometry(mesh, ids), decompose(ids)))
end

# ===============================================================================
# Mesh construction
# ===============================================================================

"""
  add_vertex!(mesh, coords::Number...)
  add_vertex!(mesh, coords::Tuple{Vararg{<:Number}})
  add_vertex!(mesh, coords::SVector{N, <: Number})

Add new vertex with the given coordinates to the mesh

```julia
# Add vertex with coordinates (0., 1.)
add_vertex!(mesh, 0., 1.)
add_vertex!(mesh, (0., 1.))
add_vertex!(mesh, SVector{3, Float64}(0., 1.))
```
"""
add_vertex!(mesh::Mesh, coords::Number...) = add_vertex!(mesh, coords)

# add vertex to mesh with world dimension one
add_vertex!(mesh::Mesh{<:Cell, 1}, coords::Tuple{<:Number}) = push!(mesh.nodes, coords[1])

# add vertex to mesh with world dimension at least two
function add_vertex!(mesh::Mesh, coords::Tuple{<:Number, <:Number, Vararg{<:Number}})
  add_vertex!(mesh, SVector{world_dim(mesh), real_type(mesh)}(coords))
end

# add vertex to mesh with coordinates given as a static vector
function add_vertex!(mesh::Mesh{<:Cell, world_dim}, coords::SVector{world_dim}) where world_dim
  @assert world_dim > 1
  @inbounds push!(mesh.nodes, coords)
end

# add vertex with prescribed id to mesh
add_vertex!(mesh::Mesh, i::Id, coords::Number...) = add_vertex!(mesh, i, coords)

# add vertex with prescribed id to mesh with world dimension one
function add_vertex!(mesh::Mesh{<:Cell, world_dim}, i::Id, coords::Tuple{<:Number}) where world_dim
  push!(mesh.nodes, i, coords[1])
end

# add vertex with prescribed id to mesh with world dimension at least two
function add_vertex!(mesh::Mesh, i::Id, coords::Tuple{<:Number, <:Number, Vararg{<:Number}})
  add_vertex!(mesh, i, SVector{world_dim(mesh), real_type(mesh)}(mesh)(coords))
end

# add vertex with prescribed id to mesh given as static vector
function add_vertex!(mesh, i::Id, coords::SVector)
  @inbounds push!(mesh.nodes, i, coords)
end

export add_vertices!
function add_vertices!(mesh, vertices::AbstractArray{REAL_, 2}) where REAL_ <: Real
  println("add vertices")
  @assert world_dim(mesh)==size(vertices, 1)
  bla=reinterpret(SVector{world_dim(mesh),REAL_}, vertices, (size(vertices, 2),))
  #println("bla", mesh.nodes, bla)
  append!(mesh.nodes,bla)
  nothing
end

function add_cell!(mesh::Mesh, ::Type{K}, vertices::VID...) where {K <: Cell, VID <: Union{Id, Int}}
  add_cell!(mesh, Connectivity{K, vertex(K)}(vertices...))
end

function add_cell!(mesh::Mesh, conn::Connectivity{K}) where K <: Cell
  @assert(K ∈ flatten(type_scatter(skeleton(element_type(mesh)))),
          """Can not add a cell of type $(K) into a mesh with element type $(element_type(msh))
          admissable cell types: $(flatten(type_scatter(skeleton(element_type(msh)))))""")
  @inbounds push!(connectivity(mesh, dim_t(K), Dim{0}()), K, conn)
end

function add_cell!(mesh::Mesh, i::Id{K}, conn::Connectivity{K}) where K <: Cell
  push!(connectivity(mesh, dim_t(K), Dim{0}()), i, conn)
end

# since the dimension can not be deduced in a generated function we have to pass
#  it explicitly as a parameter
@generated function _cells(mesh::Mesh, i::Dim, ::K) where K <: Cell
  i == Dim{0} ? :(domain(vertex_coordinates(mesh))) : :(domain(connectivity(mesh, dim_t(K), Dim{0}())[K]))
end

@generated function cells(mesh::Mesh, i::Dim)
  i == Dim{0} ? :(domain(vertex_coordinates(mesh))) : :(domain(connectivity(mesh, i, Dim{0}())))
end

cells(mesh::Mesh, ::K) where K <: Cell = _cells(mesh, dim_t(K), K())

"Return the vertex ids of the nodes"
vertices(mesh::Mesh) = domain(mesh.nodes)

"Return the vertex coordinates of the nodes of a mesh `mesh`"
vertex_coordinates(mesh::Mesh) = mesh.nodes

function vertices(mesh::Mesh, cells::HomogeneousIdIterator{K}) where K <: Cell
  domain(connectivity(mesh, K))[vertices(mesh)]
end

# look wether https://github.com/JuliaArrays/MappedArrays.jl
#  is an alternative
geometry(mesh::Mesh) = geometry(mesh, Codim{0}())

"""
    geometry(mesh::Mesh, i::Dim)

Return geometry of all `i`-dimensional cells
"""
@dim_dispatch geometry(mesh::Mesh{K}, i::Dim) where K <: Cell = geometry(mesh, cells(mesh, i))

"""
    geometry(mesh::Mesh, ::Union{C, Type{C}}) where C <: Cell

Return geometry of all cells with type `C`
"""
geometry(mesh::Mesh, ::Union{C, Type{C}}) where C <: Cell = geometry(mesh, cells(mesh, C()))

"""
  geometry(mesh::Mesh, ids)

Return geometry of all cells with ids in `ids`
"""
function geometry(mesh::M, ids::HomogeneousIdIterator{C}) where {C <: Cell, M <: Mesh}
  @assert typeof(C) == DataType "Only a single cell type allowed"
  if dim(C)==0
    return vertex_coordinates(mesh)
  else
    let vertex_coordinates = vertex_coordinates(mesh),
        mesh_conn = connectivity(mesh, C(), Dim{0}())
      mesh_geo = MeshFunction(C, Geometry{C, world_dim(M), real_type(M)}, Val{false}())
      #resize!(mesh_geo, length(ids))
      for id in ids
        push!(mesh_geo, id, map(vidx -> vertex_coordinates[vidx], mesh_conn[id]))
      end
      mesh_geo
    end
  end
end

function geometry(mesh::Mesh, ids::HeterogenousIdIterator)
  compose(map(ids -> geometry(mesh, ids), decompose(ids)))
end

"""
    geometry(mesh::Mesh, id::Id{C}) where C <: Cell

Return geometry of the cell with id `id`
"""
function geometry(mesh::Mesh, id::Id{C}) where C <: Cell
  nodes = vertex_coordinates(mesh)
  conn = vertex_connectivity(mesh, C())[id]
  Geometry{C, world_dim(mesh), real_type(mesh)}(map(vidx -> nodes[vidx], conn))
end

"""
    elements(mesh::Mesh)

Return the ids of all mesh elements (i.e. codim zero cells)
"""
elements(mesh::Mesh) = domain(connectivity(mesh, Codim{0}(), Dim{0}()))

"""
    cell_groups(mesh::Mesh)

Return a dictionary of all cell groups, where the key is the tag and the value
the corresponding cell group
"""
cell_groups(mesh::Mesh) = mesh.cell_groups

"""
    tagged_cells(mesh::Mesh, tag)

Return the cell group tagged with `tag`
"""
tagged_cells(mesh::Mesh, tag) = cell_groups(mesh)[tag]

"""
    tag_cells!(pred::Function, mesh::Mesh, ::Union{C, Type{C}}, tag) where C <: Cell

Given a predicate `pred` taking a cell geometry tag all cells for which
`pred` returns true with `tag`
"""
function tag_cells!(pred::Function, mesh::Mesh, ::Union{C, Type{C}}, tag) where C <: Cell
  cell_group = IdIterator(C)
  foreach(decompose(geometry(mesh, C))) do mesh_geo
    map(graph(mesh_geo)) do cid, geo
      if pred(cid, geo)
        push!(cell_group, cid)
      end
    end
    cell_group
  end
  cell_groups(mesh)[tag] = cell_group
end

function tag_cells!(pred::Function, mesh::Mesh, cells::TypedIdIterator, tag)
  C = cell_type(cells)
  cell_group = IdIterator(C)
  foreach(decompose(geometry(mesh, cells))) do mesh_geo
    map(graph(mesh_geo)) do cid, geo
      if pred(cid, geo)
        push!(cell_group, cid)
      end
    end
    cell_group
  end
  cell_groups(mesh)[tag] = cell_group
end

"""
Given a predicate `pred` taking a cell geometry tag all vertices for which
`pred` returns true
"""
function tag_vertices!(pred::Function, mesh::Mesh, vertex_ids::TypedIdIterator, tag)
  tagged_vertices = Vector{Id{vertex(element_type(mesh))}}()
  let vertex_coordinates = vertex_coordinates(mesh)
    for vid in vertex_ids
      if pred(vertex_coordinates[vid])
        push!(tagged_vertices, vid)
      end
    end
  end
  cell_groups(mesh)[tag] = tagged_vertices
end

function tag_vertices!(pred::Function, msh::Mesh, tag)
  tag_vertices!(pred, msh, vertices(msh), tag)
end

"""
Extract mesh that contains only the cells on the boundary
"""
function boundary(mesh::Mesh{K, world_dim, REAL_}) where {K<:Cell, world_dim, REAL_ <: Real}
  # construct a new mesh
  boundary_mesh = Mesh{facet(K), world_dim, REAL_, false, typeof(mesh)}(mesh)
  # get all boundary cells
  boundary_cells = tagged_cells(mesh, :boundary)
  # get the connectivity of all Codim{1} cells
  el_mesh_conn = connectivity(mesh, Codim{1}(), Dim{0}())
  # the number of vertices of all cells on the boundary
  N = vertex_count(facet(K))*length(boundary_cells)
  # the ids of all vertices
  vertices = Vector{Id{vertex(K)}}(undef, N)
  k=1
  for cid in boundary_cells
    el_conn = el_mesh_conn[cid]
    add_cell!(boundary_mesh, cid, el_conn)
    for v in el_conn
      vertices[k] = v
      k+=1
    end
  end
  # add nodes on the boundary to the boundary mesh
  let vertex_coordinates = vertex_coordinates(mesh)
    vertex_ids = unique(vertices)
    for vid in vertex_ids
      add_vertex!(boundary_mesh, vid, vertex_coordinates[vid])
    end
  end
  # return mesh
  boundary_mesh
end

function extract(mesh::Mesh{K, world_dim, REAL_}, cells::HomogeneousIdIterator) where {K<:Cell, world_dim, REAL_ <: Real}
  dim(cell_type(cells)) == true || error("Not implemented")

  # construct a new mesh
  extracted_mesh = Mesh{facet(K), world_dim, REAL_, false, typeof(mesh)}(mesh)
  # get the connectivity of all Codim{1} cells
  el_mesh_conn = connectivity(mesh, Codim{1}(), Dim{0}())
  # the number of vertices of all cells on the boundary
  N = vertex_count(facet(K))*length(cells)
  # the ids of all vertices
  vertices = Vector{Id{vertex(K)}}(N)
  k=1
  for cid in cells
    el_conn = el_mesh_conn[cid]
    add_cell!(extracted_mesh, cid, el_conn)
    for v in el_conn
      vertices[k] = v
      k+=1
    end
  end
  # add nodes on the boundary to the boundary mesh
  let vertex_coordinates = vertex_coordinates(mesh)
    vertex_ids = unique(vertices)
    for vid in vertex_ids
      add_vertex!(extracted_mesh, vid, vertex_coordinates[vid])
    end
  end
  # return mesh
  extracted_mesh
end

populate_connectivity!(msh::Mesh) = _populate_connectivity!(msh, Val{mesh_dim(msh)}())

function _populate_connectivity!(msh::Mesh{Ks}, mesh_dim::Val{1}) where Ks <: Cell
  # initialize connectivity from vertices → segments
  let conn_t = connectivity_type(cell_type(msh, Dim{1}()), Dim{0}(), Dim{1}())
    v2s_conn = topology(msh)[Dim{0}(), Dim{1}(), true]
    set_domain!(v2s_conn, vertices(msh))
    set_image!(v2s_conn, Vector{conn_t}(undef, length(vertices(msh))))
    fill!(image(v2s_conn), conn_t(0, 0))
    for (sid, (geo, conn)) in graph(zip(geometry(msh), vertex_connectivity(msh)))
      li, ri = geo[1] < geo[2] ? (2, 1) : (1, 2)
      v2s_conn[conn[li], 1] = sid
      v2s_conn[conn[ri], 2] = sid
    end
    mark_populated!(topology(msh), Dim{0}(), Dim{1}())
  end
  # intitialize connectivity from segments → segments
  let conn_t = connectivity_type(cell_type(msh, Dim{1}()), Dim{1}(), Dim{1}())
    s2s_conn = topology(msh)[Dim{1}(), Dim{1}(), true]
    set_domain!(s2s_conn, cells(msh, Dim{1}()))
    set_image!(s2s_conn, Vector{conn_t}(undef, length(vertices(msh))))
    fill!(image(s2s_conn), conn_t(0, 0))
    for conn in connectivity(msh, Dim{0}(), Dim{1}())
      s2s_conn[conn[1], 2] = conn[2]
      s2s_conn[conn[2], 1] = conn[1]
    end
  end
  # identify boundary cells
  #  create mesh function from vertices to number of adjacent cells
  mf = MeshFunction(vertices(msh), zeros(UInt8, length(vertices(msh))))
  #  iterate over cells/segments and increment adjacency counter for both
  #   of its vertices
  for conn in vertex_connectivity(msh)
    mf[conn[1]] += 1
    mf[conn[2]] += 1
  end
  #  filter out all vertices with only a single adjacent cell. these
  #   are the boundary cells
  boundary_cells = Id{cell_type(msh, Dim{0}())}[]
  for (id, v) in graph(mf)
    v == 1 && push!(boundary_cells, id)
  end
  #  store boundary cells
  cell_groups(msh)[:boundary] = boundary_cells
  msh
end

function _populate_connectivity!(msh::Mesh{Ks}, mesh_dim::Val{2}) where Ks <: Cell
  facet_conn_t = Connectivity{facet(Ks), subcell(Ks, Dim{0}())}
  # clear connectivity (in case the connectivity is already populated)
  clear_connectivity!(topology(msh), Codim{0}(), Codim{1}())
  clear_connectivity!(topology(msh), Codim{0}(), Codim{0}())
  clear_connectivity!(topology(msh), Codim{1}(), Dim{0}())
  # get facet mesh connectivity (incidence relation codim 1 -> dim 0)
  facets_mesh_conn = topology(msh)[Codim{1}(), Dim{0}(), true]
  # get element facet mesh connectivity (incidence relation codim 0 -> dim 1)
  el_facet_mesh_conn = topology(msh)[Codim{0}(), Codim{1}(), true]
  # add edges in the interior to the mesh
  potential_boundary_facet_tuples = Vector{Tuple{Id, Int, Bool, fulltype(facet_conn_t)}}()
  mesh_conn = connectivity(msh, Codim{0}(), Dim{0}())
  # todo: this gives a 5x performance degradation -> move to a seperate function
  #  this might be caused by julia #15276
  foreach(decompose(connectivity(msh, Codim{0}(), Dim{0}()))) do mesh_conn
    K = cell_type(mesh_conn)
    resize!(el_facet_mesh_conn[K], number_of_cells(msh, K))
    # allocate a vector containing the facet tuples
    facet_tuples = Vector{Tuple{Id{K}, Int, Bool, fulltype(facet_conn_t)}}()
    # first add all half facets
    for (cid, cell_conn) in graph(mesh_conn)
      for (local_index, facet_conn) in enumerate(facets(cell_conn))
        push!(facet_tuples, (cid, local_index,
              is_cannonical(facet_conn), canonicalize_connectivity(facet_conn)))
      end
    end
    # sort the facet tuples by the the facet connectivity
    sort!(facet_tuples, by=ft->ft[4])
    # here we store the indices in the facet tuples array of all facet tuples
    #  that potentially lie on the boundary
    potential_boundary_facet_tuple_indices = Vector{Int}()
    # now iterate over all facet tuples
    let last_pid = Id{K}(0), # parent cell id in the last iteration
        last_facet_conn = facet_conn_t(), # facet connectivity in the last iteration
        last_local_index = 0,
        i=1 # current index
      for (pid, local_index, orientation_changed, facet_conn) in facet_tuples
        if last_facet_conn == facet_conn
          pop!(potential_boundary_facet_tuple_indices)
          push!(facets_mesh_conn, facet_conn)
          facet_id = last(domain(facets_mesh_conn))
          el_facet_mesh_conn[pid, local_index] = facet_id
          el_facet_mesh_conn[last_pid, last_local_index] = facet_id
        else
          push!(potential_boundary_facet_tuple_indices, i)
        end
        # update current index and store the pid and facet_conn of the current
        #  iteration
        i+=1
        last_pid=pid
        last_local_index=local_index
        last_facet_conn=facet_conn
      end
    end
    # now add all potential boundary facet tuples
    for i in potential_boundary_facet_tuple_indices
      push!(potential_boundary_facet_tuples, facet_tuples[i])
    end
  end
  boundary_facet_ids = Vector{Id{facet(Ks)}}()
  boundary_facet_tuples = Vector{Tuple{Id, Int, Bool, fulltype(facet_conn_t)}}()
  # sort the potential boundary facet tuples by the the facet connectivity
  sort!(potential_boundary_facet_tuples, by=ft->ft[4])
  # now iterate over all potential boundary facet tuples and remove duplicates
  last_pid = nothing # parent cell id in the last iteration
  last_local_index = 0
  last_facet_conn = facet_conn_t() # facet connectivity in the last iteration
  for (pid, local_index, orientation_changed, facet_conn) in potential_boundary_facet_tuples
    if last_facet_conn == facet_conn
      # add it to the mesh
      push!(facets_mesh_conn, facet_conn)
      facet_id = last(domain(facets_mesh_conn))
      el_facet_mesh_conn[pid, local_index] = facet_id
      el_facet_mesh_conn[last_pid, last_local_index] = facet_id
      # remove it from the boundary facet tuples
      pop!(boundary_facet_tuples)
    else
      push!(boundary_facet_tuples, (pid, local_index, orientation_changed, facet_conn))
    end
    last_pid=pid
    last_local_index=local_index
    last_facet_conn=facet_conn
  end
  # now that we have found all boundary facet tuples add them to the mesh
  for (pid, local_index, orientation_changed, facet_conn) in boundary_facet_tuples
    push!(facets_mesh_conn, orientation_changed ? flip_orientation(facet_conn) : facet_conn)
    facet_id = last(domain(facets_mesh_conn))
    el_facet_mesh_conn[pid, local_index] = facet_id
    push!(boundary_facet_ids, facet_id)
  end
  # sanity check
  @assert(mapreduce(Ks -> facet_count(Ks)*number_of_cells(msh, Ks),
    +, cell_types(msh))+length(boundary_facet_ids) == 2*length(facets_mesh_conn),
    "mesh topology integrity check 1 failed")
  # mark connectivities as populated
  mark_populated!(topology(msh), Codim{0}(), Codim{1}())
  mark_populated!(topology(msh), Codim{0}(), Codim{0}())
  mark_populated!(topology(msh), Codim{1}(), Dim{0}())
  # add boundary edge markers to the mesh
  cell_groups(msh)[:boundary] = boundary_facet_ids
  msh
end

#function populate_connectivity!(msh::Mesh{Ks}) where Ks <: Cell
#  const facet_conn_t = Connectivity{facet(Ks), subcell(Ks, Dim{0}())}
#  # clear connectivity (in case the connectivity is already populated)
#  clear_connectivity!(topology(msh), Codim{0}(), Codim{1}())
#  clear_connectivity!(topology(msh), Codim{0}(), Codim{0}())
#  clear_connectivity!(topology(msh), Codim{1}(), Dim{0}())
#  # get facet mesh connectivity (incidence relation codim 1 -> dim 0)
#  facets_mesh_conn = topology(msh)[Codim{1}(), Dim{0}(), true]
#  # number of half facets
#  N = mapreduce(K -> facet_count(K)*number_of_cells(msh, K), +, uniontypes(Ks))
#  # allocate vector containing the connectivity of all facets
#  cannonical_half_facets_conn =  # todo: preallocate
#  potential_boundary_facets = Vector{Tuple{Id, fulltype(facet_conn_t)}}()
#  # ...
#  foreach(decompose(connectivity(msh, Codim{0}(), Dim{0}()))) do mesh_conn
#    K = element_type(mesh_conn)
#    # get all half facets from the codim 0 cells of type K
#    cannonical_half_facets_conn = Vector{Tuple{Id{K}, fulltype(facet_conn_t)}}()
#    for (cid, cell_conn) in graph(mesh_conn)
#      for facet_conn in facets(cell_conn)
#        push!(cannonical_half_facets_conn, (cid, canonicalize_connectivity(facet_conn)))
#      end
#    end
#    # sort
#    sort!(cannonical_half_facets_conn)
#    last_pid = Id{K}(0)
#    last_facet_conn = facet_conn_t()
#    for (pid, facet_conn) in cannonical_half_facets_conn
#      # if the last facet was not a boundary facet
#      if facet_conn == last_facet_conn
#        # therefore reomove it from the potential boundary facets
#        pop!(potential_boundary_facets)
#        # and add it to the facet mesh connectivity
#        push!(facets_mesh_conn, facet_conn)
#        codim0_codim1_mesh_conn[pid] = modify(codim0_codim1_mesh_conn[pid],
#          last(domain(facets_mesh_conn)), lidx)
#        push!(codim0_codim1_mesh_conn, modify(lidx, ))
#      else
#        push!(potential_boundary_facets, (last_parent_cid, facet_conn))
#      end
#      last_pid = pid
#      last_facet_conn = facet_conn
#    end
#  end
#  sort!(potential_boundary_facets, by=x->x[2])
#  boundary_facet_ids = Array{Id{facet(Ks)}, 1}()
#  last_facet_conn = facet_conn_t()
#  for (pid, facet_conn) in potential_boundary_facets
#    if facet_conn == last_facet_conn
#      pop!(boundary_facet_ids)
#    else
#      push!(facets_mesh_conn, facet_conn)
#      push!(boundary_facet_ids, domain(facets_mesh_conn)[end])
#    end
#    last_facet_conn = facet_conn
#  end
#  # sanity check
#  @assert(mapreduce(Ks -> facet_count(Ks)*number_of_cells(msh, Ks),
#    +, cell_types(msh))+length(boundary_facet_ids) == 2*length(facets_mesh_conn),
#    "mesh topology integrity check 1 failed")
#  # mark connectivities as populated
#  mark_populated!(topology(msh), Codim{0}(), Codim{1}())
#  mark_populated!(topology(msh), Codim{0}(), Codim{0}())
#  mark_populated!(topology(msh), Codim{1}(), Dim{0}())
#  # add boundary edge markers to the mesh
#  cell_groups(msh)[:boundary] = boundary_facet_ids
#  #end
#  msh
#end

#"""
#Populate mesh topology (restricted bidirectional)
#"""
#function populate_connectivity!(msh::Mesh{K}) where K <: Cell
#  const edge_conn_t = Connectivity{facet(K), subcell(K, Dim{0}())}
#  # clear connectivity (in case the connectivity is already populated)
#  clear_connectivity!(topology(msh), Codim{0}(), Codim{1}())
#  clear_connectivity!(topology(msh), Codim{0}(), Codim{0}())
#  clear_connectivity!(topology(msh), Codim{1}(), Dim{0}())
#  # allocate a temporary mesh function for the mesh connectivity of the boundary
#  #  edges
#  tmp_boundary_edges = MeshFunction(facet(K), edge_conn_t)
#  # get edge mesh connectivity (incidence relation codim 1 -> dim 0)
#  edges = topology(msh)[Codim{1}(), Dim{0}(), true]
#  # sanity check
#  @assert typeof(tmp_boundary_edges) == typeof(edges) "Type instability detected"
#  # populate connectivity
#  foreach(decompose(connectivity(msh, Codim{0}(), Dim{0}()))) do el_conn
#    let K = cell_type(el_conn)
#      # retrieve all edges from the facets
#      # - every edge occurs twice, but with a different parent triangle
#      #   which is encoded in the index of the flattened mesh function
#      # - the edges have no identity
#      ul_edges = flatten(map(facets, el_conn))
#      # store local indices
#      ul_edges_lidx = MeshFunction(domain(ul_edges), repeat(1:vertex_count(K), outer=length(el_conn)))
#      # allocate mesh function for the incidence relation codim 0 -> codim 1
#      el_facets = MeshFunction(K, Connectivity{K, subcell(K)})
#      resize!(el_facets, length(el_conn), ((-1 for i in 1:vertex_count(K))...))
#      # allocate mesh function for the neighbourhood relation
#      neighbours = MeshFunction(K, NeighbourConnectivity{K})
#      resize!(neighbours, length(el_conn), ((-1 for i in 1:vertex_count(K))...))
#      # sort unlabeled edges by vertex list
#      A = sort(zip(ul_edges_lidx, ul_edges), by=(x)->x[2])
#      # sort unlabeled edges by vertex list reversed
#      B = sort(zip(ul_edges_lidx, ul_edges), by=x->reverse(x[2]))
#
#      # stop early if there are no edges (i.e. empty mesh)
#      if length(A)>0
#        i_b = 1
#        for (i_a, (il_a, a)) in graph(A)
#          # until we find an edge with opposite direction we assume that `a` is
#          #  on the boundary
#          is_on_boundary = true
#          il_b, b = B[i_b]
#          # higher i_b until it points to an edge ending in source of `a`
#          while a[1] > b[2] && i_b < length(B)
#            i_b+=1
#            il_b, b = B[i_b]
#          end
#          # increase i_b until we find either find the opposing edge or i_b points
#          #  to an edge whose end is not equal to the source of `a` anymore.
#          # note that all edges which are skipped by increasing i_b are
#          #  boundary edges
#          while a[1] == b[2] && a[2] >= b[1]
#            if a[1] == b[2] && a[2] == b[1]
#              # save neighbourhood relation between codim 0 cells
#              neighbours[i_a, il_a] = domain(B)[i_b]
#              # mark edge as a non boundary edge
#              is_on_boundary = false
#              i_b+=1
#              break
#            end
#            i_b+=1
#            if i_b < length(B)
#              il_b, b = B[i_b]
#            else
#              break
#            end
#          end
#          # if the current edge is a boundary edge we just add it to the edge mesh
#          # function. if the edge is not on the boundary we only add it if the index
#          #  of its source is higher then the index of its sink. This ensures
#          #  the we add every edge only once.
#          if is_on_boundary || a[1]<a[2]
#            let edges = is_on_boundary ? tmp_boundary_edges : edges
#              push!(edges, a)
#              el_facets[i_a, il_a] = last(domain(edges))
#            end
#          end
#          assert(i_b <= length(ul_edges)+1)
#        end
#        #assert(length(edges)%2==0)
#        # throw away all duplicate edges
#        #resize!(edges, Integer(length(edges)/2))
#      end
#      # write back facets and neighbours
#      topology(msh)[Codim{0}(), Codim{1}(), true][K] = el_facets
#      topology(msh)[Codim{0}(), Codim{0}(), true][K] = neighbours
#    end
#  end
#  # remove duplicate boundary edges (occurs for edges between two elements
#  #  of different cell type)
#  boundary_edge_count = 0
#  boundary_edge_ids = Array{Id{facet(K)}, 1}()
#  last_edge = edge_conn_t(-1, -1)
#  for edge in sort(tmp_boundary_edges, by=canonicalize_connectivity)
#    edge = canonicalize_connectivity(edge)
#    # add edge if it differs from the last edge
#    if edge != last_edge
#      boundary_edge_count+=1
#      push!(edges, edge)
#      push!(boundary_edge_ids, last(domain(edges)))
#    else
#      # remove the last edge because it is no boundary edge
#      pop!(boundary_edge_ids)
#      boundary_edge_count-=1
#    end
#    last_edge = edge
#  end
#  # sanity check
#  let edges=topology(msh)[Codim{1}(), Dim{0}(), true]
#    println(boundary_edge_count)
#    @assert(mapreduce(K -> facet_count(K)*number_of_cells(msh, K),
#                     +, cell_types(msh))+boundary_edge_count == 2*length(edges),geometry
#            "mesh topology integrity check 1 failed")
#    @assert(length(boundary_edge_ids) == boundary_edge_count,
#            "mesh topology integrity check 2 failed")
#  end
#  # mark connectivities as populated
#  mark_populated!(topology(msh), Codim{0}(), Codim{1}())
#  mark_populated!(topology(msh), Codim{0}(), Codim{0}())
#  mark_populated!(topology(msh), Codim{1}(), Dim{0}())
#  # add boundary edge markers to the mesh
#  cell_groups(msh)[:boundary] = boundary_edge_ids
#end

canonicalize_connectivity(::MethodNotImplemented) = error("Method not implemented")

#@generated function vertices{K, M}(c::CellRef{K, M})
#  :(SVector{$(vertex_count(K)), $(CellRef{vertex(K),M})}((CellRef(mesh(c), idx) for idx in connectivity(mesh(c), Dim{dim(K)}(), Dim{0}())[index(c)])...))
#end

#function vertex(c::CellRef, i::Int)
#  vertices(c)[i]
#end

#@generated function vertices{T <: Mesh}(mesh::T)
#  let K = cell_type(T), vt = vertex(K)
#    :(MeshFunction(Id{$(vt)}(1):Id{$(vt)}(length(mesh.coordinates)),
#                   (collect($(vt)(c...) for c in mesh.coordinates))))
#  end
#end

#cell_types(mesh::Mesh) = Set(vcat((isnull(c) ? [] : cell_types(c) for c in mesh.connectivity)...))
