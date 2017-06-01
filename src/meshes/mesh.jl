using Iterators
using StaticArrays
using SimpleRepeatIterator
using TipiFEM.Utils: type_scatter, flatten
export subcell, world_dim

# todo: add invalid flag to all cell entities

using Base.OneTo
"""
Construct a generic mesh containing `K` cells.

```julia
Mesh(Polygon"3-node triangle")
Mesh(Union{Polygon"3-node triangle", Polygon"4-node quadrangle"})
```
"""
@computed immutable Mesh{K, world_dim, REAL_ <: Real, simple, PM}
  nodes::HomogenousMeshFunction{vertex(K), SVector{world_dim, REAL_},
    simple ? OneTo{Index{vertex(K)}} : Vector{Index{vertex(K)}}, Vector{SVector{world_dim, REAL_}}}
  topology::fulltype(MeshTopology{K, simple})
  attributes::Dict{Any, MeshFunction}
  cell_groups::Dict{Any, Vector{Index}}
  parent::PM

  function Mesh(parent::PM=nothing)
    mesh = new{K, world_dim, REAL_, simple, PM}(
      MeshFunction(vertex(K), SVector{world_dim, REAL_}, Val{simple}()),
      MeshTopology(K, Val{simple}()), Dict{Symbol, MeshFunction}(), Dict{Any, Array{Cell, 1}}(),
      parent)
    mark_populated!(mesh.topology, dim_t(K), Dim{0}())
    #attributes(msh)[:proper] = true
    mesh
  end
end

Mesh{K <: Cell}(::Type{K}; simple=Val{true}) = Mesh{K, dim(K), Float64, simple==Val{true}, Void}()

"is the mesh simple"
@typeinfo issimple(M::Type{<:Mesh}) =  tparam(M, 4)

"get type of the parent mesh"
@typeinfo parent_type(M::Type{<:Mesh}) = tparam(M, 5)

"does the mesh has a parent"
@typeinfo hasparent(M::Type{<:Mesh}) = parent_type(M) != Void

import Base.parent

parent(mesh::Mesh) = mesh.parent

"dimension of the ambient space"
@typeinfo world_dim(M::Type{<:Mesh}) = tparam(M, 2)

"type of cells in `M` with (co)-dimension `d`"
@typeinfo cell_type(M::Type{<:Mesh}, d=Codim{0}()) = let K = tparam(M, 1)
  subcell(K, d)
end

"type of the mesh function mapping node indices to node coordinates"
@typeinfo nodes_type{M <: Mesh}(::Type{M}) = fieldtype(M, :nodes)

"type of cells in `M` with (co)-dimension `d` as a tuple"
@typeinfo cell_types(M::Type{<:Mesh}, d=Codim{0}()) = (Base.uniontypes(cell_type(M, d))...)

"Number of `d` dimensional cells in `msh`"
@dim_dispatch function number_of_cells{K}(msh::Mesh{K}, d::Codim)
  # if the number of vertices is requested we have to look at the nodes arrays
  if complement(cell_type(msh), d)==Dim{0}()
    length(msh.nodes)
  # otherwise the number of cells is stored in the topology array
  else
    ispopulated(topology(msh), d, Dim{0}()) ? length(connectivity(msh, d, Dim{0}())) : 0
  end
end

"Number of cells with codimension 0"
number_of_cells(msh::Mesh) = number_of_cells(msh, Codim{0}())

"Number of cells of type `T` in `msh`"
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

function dumpmesh(msh::Mesh{K}) where K <: Cell
  # show the usual mesh information
  display(msh)
  # show all nodes
  println("  Nodes: ")
  for (nid, coordinates) in graph(nodes(msh))
    println("    $(nid) ↦ $(coordinates)")
  end
  println()
  # show all cells
  for d in dim(K):-1:1
    println("  $(d) dimensional cells:")
    for (cid, cell_conn) in graph(connectivity(msh, Dim{d}(), Dim{0}()))
      println("   $(cid) ↦ $(cell_conn)")
    end
    println()
  end
end

"dimension of codimension zero cells"
@typeinfo mesh_dim(M::Type{<:Mesh}) = dim(cell_type(M))

"type used for calculations with real numbers"
@typeinfo real_type(M::Type{<:Mesh}) = tparam(M, 3)

topology(mesh) = mesh.topology

function set_connectivity!(mesh, d1, d2, mf::MeshFunction)
  # todo: check type of the mesh function
  mesh.topology[Dim{d1}(), Dim{d2}()] = mf
end

"""
connectivity of i-cells with j-cells

this represents the incedence relation of i-cells with j-cells
"""
@dim_dispatch function connectivity{K <: Cell, i, j}(mesh::Mesh{K}, d1::Dim{i}, d2::Dim{j})
  topology(mesh)[d1, d2]
end

@dim_dispatch function connectivity{K <: Cell, C <: Cell}(mesh::Mesh{K}, ::C, j::Dim)
  #assert(subcell(C1, dim_t(C2)) == C2)
  connectivity(mesh, dim_t(C), j)[C]
end

function vertex_connectivity(mesh::Mesh, ::C) where C <: Cell
  connectivity(mesh, C(), subcell(C, Dim{0}())())
end

"add vertex with the given coordinates to the mesh"
add_vertex!(mesh, coords::Number...) = add_vertex!(mesh, SVector{world_dim(mesh), real_type(mesh)}(coords))

function add_vertex!(mesh, coords::SVector)
  @inbounds push!(mesh.nodes, coords)
end

add_vertex!(mesh, i::Index, coords::Number...) = add_vertex!(mesh, i, SVector{world_dim(mesh), real_type(mesh)}(coords))

function add_vertex!(mesh, i::Index, coords::SVector)
  @inbounds push!(mesh.nodes, i, coords)
end

export add_vertices!
function add_vertices!{REAL_ <: Real}(mesh, vertices::AbstractArray{REAL_, 2})
  println("add vertices")
  @assert world_dim(mesh)==size(vertices, 1)
  bla=reinterpret(SVector{world_dim(mesh),REAL_}, vertices, (size(vertices, 2),))
  #println("bla", mesh.nodes, bla)
  append!(mesh.nodes,bla)
  nothing
end

function add_cell!(mesh::Mesh, ::Type{K}, vertices::VID...) where {K <: Cell, VID <: Union{Index, Int}}
  add_cell!(mesh, Connectivity{K, vertex(K)}(vertices...))
end

function add_cell!{K <: Cell}(mesh::Mesh, conn::Connectivity{K})
  @assert(K ∈ flatten(type_scatter(skeleton(cell_type(mesh)))),
          """Can not add a cell of type $(K) into a mesh with element type $(cell_type(msh))
          admissable cell types: $(flatten(type_scatter(skeleton(cell_type(msh)))))""")
  @inbounds push!(connectivity(mesh, dim_t(K), Dim{0}()), K, conn)
end

function add_cell!(mesh::Mesh, i::Index{K}, conn::Connectivity{K}) where {K <: Cell}
  push!(cells(mesh, dim_t(K)), i, conn)
end

function cells(mesh::Mesh, ::K) where K <: Cell
  dim(K) == 0 ? domain(nodes(mesh)) : domain(connectivity(mesh, dim_t(K), Dim{0}())[K])
end

function cells(mesh::Mesh, i::Dim)
  i == Dim{0}() ? domain(nodes(mesh)) : domain(connectivity(mesh, i, Dim{0}()))
end

import Base.eltype

"Return the vertex ids of the nodes"
vertices(mesh::Mesh) = domain(mesh.nodes)

"Return the coordinates of the nodes of a mesh `mesh`"
nodal_coordinates(mesh::Mesh) = image(mesh.nodes)

"Return the nodes of a mesh `mesh`"
nodes(mesh::Mesh) = mesh.nodes

function vertices(mesh::Mesh, cells::IdIterator{K}) where K <: Cell
  domain(connectivity(mesh, K))[vertices(mesh)]
end

# look wether https://github.com/JuliaArrays/MappedArrays.jl
#  is an alternative
geometry(mesh::Mesh) = geometry(mesh, Codim{0}())

@dim_dispatch function geometry{K <: Cell}(mesh::Mesh{K}, i::Dim)
  mapreduce(chain, decompose(connectivity(mesh, i, Dim{0}()))) do mesh_conn
    const geo_t = Geometry{cell_type(mesh_conn), world_dim(mesh), real_type(mesh)}
    map(graph(mesh_conn)) do idx, conn
      geo_t(map(vidx -> nodes(mesh)[vidx], conn))
    end
  end
end

function geometry(mesh::Mesh, ::Union{C, Type{C}}) where {C <: Cell}
  mesh_conn = connectivity(mesh, C(), Dim{0}())
  const geo_t = Geometry{cell_type(mesh_conn), world_dim(mesh), real_type(mesh)}
  map(graph(mesh_conn)) do idx, conn
    geo_t(map(vidx -> nodes(mesh)[vidx], conn))
  end
end

function geometry(mesh::M, ids::Range{Index{C}}) where {M <: Mesh, C <: Cell}
  @assert typeof(C) == DataType "Only a single cell type allowed"
  let nodes = nodes(mesh),
      mesh_conn = vertex_connectivity(mesh, C())
    mesh_geo = MeshFunction(C, Geometry{C, world_dim(M), real_type(M)})
    resize!(mesh_geo, length(ids))
    for (i, id) in enumerate(ids)
      mesh_geo[i] = map(vidx -> nodes[vidx], mesh_conn[id])
    end
    mesh_geo
  end
end

function geometry(mesh::Mesh, id::Index{C}) where C <: Cell
  nodes = nodal_coordinates(mesh)
  conn = vertex_connectivity(mesh, C())[id]
  Geometry{C, world_dim(mesh), real_type(mesh)}(map(vidx -> nodes[vidx], conn))
end

"""
Return all mesh elements, i.e. the mesh connectivity from co-dim zero cells
to vertices
"""
elements(mesh) = connectivity(mesh, Codim{0}(), Dim{0}())

"""
Return a dictionary of all cell groups, where the key is the tag and the value
the corresponding cell group
"""
cell_groups(mesh::Mesh) = mesh.cell_groups

"""
Return the cell group tagged with `tag`
"""
tagged_cells(mesh::Mesh, tag) = cell_groups(mesh)[tag]

"""
Given a predicate `pred` taking a cell geometry tag all cells for which
`pred` return true
"""
function tag_cells!(pred::Function, mesh::Mesh, ::Union{C, Type{C}}, tag) where C <: Cell
  cell_group = Vector{Index}()
  foreach(decompose(geometry(mesh, C))) do mesh_geo
    map(graph(mesh_geo)) do idx, geo
      if pred(geo)
        push!(cell_group, idx)
      end
    end
    cell_group
  end
  cell_groups(mesh)[tag] = cell_group
end

"""
Given a predicate `pred` taking a cell geometry tag all cells for which
`pred` return true
"""
function tag_nodes(pred::Function, mesh::Mesh, node_ids::Array{Index}, tag)
  tagged_nodes = Array{Index{vertex_type(cell_type(mesh))}}()
  let nodes = nodes(mesh)
    for node_id in node_ids
      if pred(nodes[node_id])
        push!(tagged_nodes, node_id)
      end
    end
  end
  cell_groups(mesh)[tag] = tagged_nodes
end

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
  vertices = Vector{Index{vertex(K)}}(N)
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
  let nodes = nodes(mesh)
    vertex_ids = unique(vertices)
    for vid in vertex_ids
      add_vertex!(boundary_mesh, vid, nodes[vid])
    end
  end
  # return mesh
  boundary_mesh
end

function populate_connectivity!(msh::Mesh{Ks}) where Ks <: Cell
  const facet_conn_t = Connectivity{facet(Ks), subcell(Ks, Dim{0}())}
  # clear connectivity (in case the connectivity is already populated)
  clear_connectivity!(topology(msh), Codim{0}(), Codim{1}())
  clear_connectivity!(topology(msh), Codim{0}(), Codim{0}())
  clear_connectivity!(topology(msh), Codim{1}(), Dim{0}())
  # get facet mesh connectivity (incidence relation codim 1 -> dim 0)
  facets_mesh_conn = topology(msh)[Codim{1}(), Dim{0}(), true]
  # get element facet mesh connectivity (incidence relation codim 0 -> dim 1)
  el_facet_mesh_conn = topology(msh)[Codim{0}(), Codim{1}(), true]
  # add edges in the interior to the mesh
  potential_boundary_facet_tuples = Vector{Tuple{Index, Int, fulltype(facet_conn_t)}}()
  mesh_conn = connectivity(msh, Codim{0}(), Dim{0}())
  # todo: this gives a 5x performance degradation -> move to a seperate function
  #  this might be caused by julia #15276
  foreach(decompose(connectivity(msh, Codim{0}(), Dim{0}()))) do mesh_conn
    K = cell_type(mesh_conn)
    resize!(el_facet_mesh_conn[K], number_of_cells(msh, K))
    # allocate a vector containing the facet tuples
    facet_tuples = Vector{Tuple{Index{K}, Int, fulltype(facet_conn_t)}}()
    # first add all half facets
    for (cid, cell_conn) in graph(mesh_conn)
      for (local_index, facet_conn) in enumerate(facets(cell_conn))
        push!(facet_tuples, (cid, local_index, canonicalize_connectivity(facet_conn)))
      end
    end
    # sort the facet tuples by the the facet connectivity
    sort!(facet_tuples, by=ft->ft[3])
    # here we store the indices in the facet tuples array of all facet tuples
    #  that potentially lie on the boundary
    potential_boundary_facet_tuple_indices = Vector{Int}()
    # now iterate over all facet tuples
    let last_pid = Index{K}(0), # parent cell id in the last iteration
        last_facet_conn = facet_conn_t(), # facet connectivity in the last iteration
        i=1 # current index
      for (pid, local_index, facet_conn) in facet_tuples
        if last_facet_conn == facet_conn
          pop!(potential_boundary_facet_tuple_indices)
          push!(facets_mesh_conn, facet_conn)
          facet_id = last(domain(facets_mesh_conn))
          el_facet_mesh_conn[pid, local_index] = facet_id
        else
          push!(potential_boundary_facet_tuple_indices, i)
        end
        # update current index and store the pid and facet_conn of the current
        #  iteration
        i+=1
        last_pid=pid
        last_facet_conn=facet_conn
      end
    end
    # now add all potential boundary facet tuples
    for i in potential_boundary_facet_tuple_indices
      push!(potential_boundary_facet_tuples, facet_tuples[i])
    end
  end
  boundary_facet_ids = Vector{Index{facet(Ks)}}()
  boundary_facet_tuples = Vector{Tuple{Index, Int, fulltype(facet_conn_t)}}()
  # sort the potential boundary facet tuples by the the facet connectivity
  sort!(potential_boundary_facet_tuples, by=ft->ft[3])
  # now iterate over all potential boundary facet tuples and remove duplicates
  last_pid = nothing # parent cell id in the last iteration
  last_local_index = 0
  last_facet_conn = facet_conn_t() # facet connectivity in the last iteration
  for (pid, local_index, facet_conn) in potential_boundary_facet_tuples
    if last_facet_conn == facet_conn
      # add it to the mesh
      push!(facets_mesh_conn, facet_conn)
      facet_id = last(domain(facets_mesh_conn))
      el_facet_mesh_conn[pid, local_index] = facet_id
      el_facet_mesh_conn[last_pid, last_local_index] = facet_id
      # remove it from the boundary facet tuples
      pop!(boundary_facet_tuples)
    else
      push!(boundary_facet_tuples, (pid, local_index, facet_conn))
    end
    last_pid=pid
    last_local_index=local_index
    last_facet_conn=facet_conn
  end
  # now that we have found all boundary facet tuples add them to the mesh
  for (pid, local_index, facet_conn) in boundary_facet_tuples
    push!(facets_mesh_conn, facet_conn)
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
#  potential_boundary_facets = Vector{Tuple{Index, fulltype(facet_conn_t)}}()
#  # ...
#  foreach(decompose(connectivity(msh, Codim{0}(), Dim{0}()))) do mesh_conn
#    K = cell_type(mesh_conn)
#    # get all half facets from the codim 0 cells of type K
#    cannonical_half_facets_conn = Vector{Tuple{Index{K}, fulltype(facet_conn_t)}}()
#    for (cid, cell_conn) in graph(mesh_conn)
#      for facet_conn in facets(cell_conn)
#        push!(cannonical_half_facets_conn, (cid, canonicalize_connectivity(facet_conn)))
#      end
#    end
#    # sort
#    sort!(cannonical_half_facets_conn)
#    last_pid = Index{K}(0)
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
#  boundary_facet_ids = Array{Index{facet(Ks)}, 1}()
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
#  boundary_edge_ids = Array{Index{facet(K)}, 1}()
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
#                     +, cell_types(msh))+boundary_edge_count == 2*length(edges),
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
#    :(MeshFunction(Index{$(vt)}(1):Index{$(vt)}(length(mesh.coordinates)),
#                   (collect($(vt)(c...) for c in mesh.coordinates))))
#  end
#end

#cell_types(mesh::Mesh) = Set(vcat((isnull(c) ? [] : cell_types(c) for c in mesh.connectivity)...))
