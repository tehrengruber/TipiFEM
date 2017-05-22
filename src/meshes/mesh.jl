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
@computed immutable Mesh{K, world_dim, REAL_ <: Real}
  nodes::HomogenousMeshFunction{vertex_type(K), SVector{world_dim, REAL_},
    OneTo{Index{vertex_type(K)}}, Array{SVector{world_dim, REAL_}, 1}}
  topology::fulltype(MeshTopology{K})
  attributes::Dict{Any, MeshFunction}
  cell_groups::Dict{Any, Union{Array{Index, 1}, Chain}}

  function Mesh()
    mesh = new{K, world_dim, REAL_}(MeshFunction(vertex_type(K), SVector{world_dim, REAL_}),
                MeshTopology(K), Dict{Symbol, MeshFunction}(), Dict{Any, Array{Cell, 1}}())
    mark_populated!(mesh.topology, dim_t(K), Dim{0}())
    #attributes(msh)[:proper] = true
    mesh
  end
end

Mesh{K <: Cell}(::Type{K}) = Mesh{K, dim(K), Float64}()

"dimension of the ambient space"
@typeinfo world_dim(M::Type{<:Mesh}) = tparam(M, 2)

"type of cells in `M` with (co)-dimension `d`"
@typeinfo cell_type(M::Type{<:Mesh}, d=Codim{0}()) = let K = tparam(M, 1)
  subcell(K, d)
end

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

function connectivity{C1 <: Cell, C2 <: Cell}(mesh, ::C1, ::C2)
  #assert(subcell(C1, dim_t(C2)) == C2)
  connectivity(mesh, dim_t(C1), dim_t(C2))[C1]
end

function vertex_connectivity(mesh::Mesh, ::C) where C <: Cell
  connectivity(mesh, C(), subcell(C, Dim{0}())())
end

"add vertex with the given coordinates to the mesh"
function add_vertex!(mesh, coords...)
  const vertex_t = SVector{world_dim(mesh), real_type(mesh)}
  @inbounds push!(mesh.nodes, vertex_t(coords...))
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

function add_cell!{K <: Cell}(msh::Mesh, ::Type{K}, vertices...)
  @assert(K ∈ flatten(type_scatter(skeleton(cell_type(msh)))),
          """Can not add a cell of type $(K) into a mesh with element type $(cell_type(msh))
          admissable cell types: $(flatten(type_scatter(skeleton(cell_type(msh)))))""")
  @inbounds push!(connectivity(msh, Dim{dim(K)}(), Dim{0}()), K,
                  Connectivity{K, vertex_type(K)}(vertices...))
end

add_cell!{K <: Cell}(mesh, conn::Connectivity{K}) = push!(connectivity(mesh, Dim{dim(K)}(), Dim{0}()), conn)

import Base.eltype

vertices(mesh) = domain(mesh.nodes)

coordinates(mesh) = image(mesh.nodes)

# look wether https://github.com/JuliaArrays/MappedArrays.jl
#  is an alternative
function geometry(mesh::Mesh)
  mapreduce(chain, decompose(connectivity(mesh, Codim{0}(), Dim{0}()))) do mesh_conn
    geo_t = Geometry{cell_type(mesh_conn), world_dim(mesh), real_type(mesh)}
    map(graph(mesh_conn)) do idx, conn
      geo_t(map(vidx -> coordinates(mesh)[vidx], conn))
    end
  end
end

function geometry(mesh::Mesh, id::Index{C}) where C <: Cell
  nodes = coordinates(mesh)
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
function tag_cells(pred::Function, mesh::Mesh, tag)
  cell_group = mapreduce(chain, decompose(geometry(mesh))) do mesh_geo
    homogenous_cell_group = Array{idxtype(mesh_geo), 1}()
    map(graph(mesh_geo)) do idx, geo
      if pred(geo)
        push!(homogenous_cell_group, idx)
      end
    end
    homogenous_cell_group
  end
  cell_groups(mesh)[tag] = cell_group
end

"""
Populate mesh topology (restricted bidirectional)
"""
function populate_connectivity!(msh::Mesh{K}) where K <: Cell
  # clear connectivity (in case the connectivity is already populated)
  clear_connectivity!(topology(msh), Codim{0}(), Codim{1}())
  clear_connectivity!(topology(msh), Codim{0}(), Codim{0}())
  clear_connectivity!(topology(msh), Codim{1}(), Dim{0}())
  # boundary edges
  boundary_edges = Array{Index{facet(K)}, 1}()
  # number of boundary edges
  boundary_edge_count = 0
  # populate connectivity
  foreach(decompose(connectivity(msh, Codim{0}(), Dim{0}()))) do el_conn
    let K = cell_type(el_conn)
      # retrieve all edges from the facets
      # - every edge occurs twice, but with a different parent triangle
      #   which is encoded in the index of the flattened mesh function
      # - the edges have no identity
      ul_edges = flatten(map(facets, el_conn))
      # store local indices
      ul_edges_lidx = MeshFunction(domain(ul_edges), repeat(1:vertex_count(K), outer=length(el_conn)))
      # allocate an array for the edges (incidence relation codim 1 -> dim 0)
      edges = topology(msh)[Codim{1}(), Dim{0}(), true]
      # allocate an array for the incidence relation codim 0 -> codim 1
      el_facets = MeshFunction(K, Connectivity{K, subcell(K)})
      resize!(el_facets, length(el_conn), ((-1 for i in 1:vertex_count(K))...))
      # allocate an array for the neighbourhood relation
      neighbours = MeshFunction(K, NeighbourConnectivity{K})
      resize!(neighbours, length(el_conn), ((-1 for i in 1:vertex_count(K))...))
      # sort unlabeled edges by vertex list
      A = sort(zip(ul_edges_lidx, ul_edges), by=(x)->x[2])
      # sort unlabeled edges by vertex list reversed
      B = sort(zip(ul_edges_lidx, ul_edges), by=x->reverse(x[2]))

      # stop early if there are no edges (i.e. empty mesh)
      if length(A)>0
        i_b = 1
        for (i_a, (il_a, a)) in graph(A)
          # until we find an edge with opposite direction we assume that `a` is
          #  on the boundary
          is_on_boundary = true
          il_b, b = B[i_b]
          # higher i_b until it points to an edge ending in source of `a`
          while a[1] > b[2] && i_b < length(B)
            i_b+=1
            il_b, b = B[i_b]
          end
          # increase i_b until we find either find the opposing edge or i_b points
          #  to an edge whose end is not equal to the source of `a` anymore.
          # note that all edges which are skipped by increasing i_b are
          #  boundary edges
          while a[1] == b[2] && a[2] >= b[1]
            if a[1] == b[2] && a[2] == b[1]
              # save neighbourhood relation between codim 0 cells
              neighbours[i_a, il_a] = domain(B)[i_b]
              # mark edge as a non boundary edge
              is_on_boundary = false
              i_b+=1
              break
            end
            i_b+=1
            if i_b < length(B)
              il_b, b = B[i_b]
            else
              break
            end
          end
          # if the current edge is a boundary edge we just add it to the edge array
          # if the edge is not on the boundary we only add it if the index
          #  of its source is higher then the index of its sink. This ensures
          #  the we add every edge only once.
          if is_on_boundary || a[1]<a[2]
            push!(edges, a)
            el_facets[i_a, il_a] = last(domain(edges))
            # save that this edge is a boundary edge
            if is_on_boundary
              push!(boundary_edges, last(domain(edges)))
            end
          end
          if is_on_boundary
            boundary_edge_count+=1
          end
          assert(i_b <= length(ul_edges)+1)
        end
        #assert(length(edges)%2==0)
        # throw away all duplicate edges
        #resize!(edges, Integer(length(edges)/2))
      end
      # write back facets and neighbours
      topology(msh)[Codim{0}(), Codim{1}(), true][K] = el_facets
      topology(msh)[Codim{0}(), Codim{0}(), true][K] = neighbours
    end
  end
  # integrity check
  let edges=topology(msh)[Codim{1}(), Dim{0}(), true]
    @assert(mapreduce(K -> facet_count(K)*number_of_cells(msh, K),
                     +, cell_types(msh))+boundary_edge_count == 2*length(edges),
            "mesh topology integrity check failed")
  end
  # mark connectivities as populated
  mark_populated!(topology(msh), Codim{0}(), Codim{1}())
  mark_populated!(topology(msh), Codim{0}(), Codim{0}())
  mark_populated!(topology(msh), Codim{1}(), Dim{0}())
  # add boundary edge markers to the mesh
  cell_groups(msh)[:boundary_edges] = boundary_edges
end

#@generated function vertices{K, M}(c::CellRef{K, M})
#  :(SVector{$(vertex_count(K)), $(CellRef{vertex_type(K),M})}((CellRef(mesh(c), idx) for idx in connectivity(mesh(c), Dim{dim(K)}(), Dim{0}())[index(c)])...))
#end

#function vertex(c::CellRef, i::Int)
#  vertices(c)[i]
#end

#@generated function vertices{T <: Mesh}(mesh::T)
#  let K = cell_type(T), vt = vertex_type(K)
#    :(MeshFunction(Index{$(vt)}(1):Index{$(vt)}(length(mesh.coordinates)),
#                   (collect($(vt)(c...) for c in mesh.coordinates))))
#  end
#end

#cell_types(mesh::Mesh) = Set(vcat((isnull(c) ? [] : cell_types(c) for c in mesh.connectivity)...))
