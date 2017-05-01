using Iterators
using StaticArrays
using TipiFEM.Utils: type_scatter, flatten
export subcell, world_dim

using Base.OneTo
"""
Construct a generic mesh containing `K` cells.

```julia
Mesh(Polygon"3-node triangle")
Mesh(Union{Polygon"3-node triangle", Polygon"4-node quadrangle"})
```
"""
@computed immutable Mesh{K, world_dim, REAL_ <: Real}
  nodes::MeshFunction{OneTo{Index{vertex_type(K)}}, Array{SVector{world_dim, REAL_}, 1}}
  topology::fulltype(MeshTopology{K})
  attributes::Array{Any, 1}
  cache::Dict{Symbol, Any}

  function Mesh()
    mesh = new{K, world_dim, REAL_}(MeshFunction(vertex_type(K), SVector{world_dim, REAL_}),
                MeshTopology{K}(), Array{Any, 1}(), Dict{Symbol, Any}())
    initialize_connectivity!(mesh.topology, dim_t(K), Dim{0}())
    mesh
  end
end

Mesh{K <: Cell}(::Type{K}) = Mesh{K, dim(K), Float64}()

"dimension of the ambient space"
@Base.pure world_dim{M <: Mesh}(::Type{M}) = M.parameters[2]
world_dim{M <: Mesh}(::M) = world_dim(M)

"type of cells with"
@Base.pure cell_type{M <: Mesh}(::Type{M}, d=Codim{0}()) = let K = M.parameters[1]
  subcell(K, d)
end
@Base.pure cell_type{M <: Mesh}(::M, d=Codim{0}()) = cell_type(M, d)

@Base.pure cell_types(msh::Mesh, d=Codim{0}()) = (Base.uniontypes(cell_type(msh, d))...)

"Number of `d` dimensional cells in `msh`"
@dim_dispatch function number_of_cells{K}(msh::Mesh{K}, d::Codim)
  # if the number of vertices is requested we have to look at the nodes arrays
  if complement(cell_type(msh), d)==Dim{0}()
    length(msh.nodes)
  # otherwise the number of cells is stored in the topology array
  else
    hasindex(topology(msh), d, Dim{0}()) ? length(connectivity(msh, d, Dim{0}())) : 0
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
    hasindex(topology(msh), dim_t(T), Dim{0}()) ||
      error("number of cells is not known yet. Did you forget to call populate_connectivity!?")
    # return the number of cells
    length(connectivity(msh, dim_t(T), Dim{0}())[T])
  end
end

"dimension of codimension zero cells"
@Base.pure mesh_dim{M <: Mesh}(::Type{M}) = dim(cell_type(M))
mesh_dim{M <: Mesh}(::M) = mesh_dim(M)

"type used for calculations with real numbers"
@Base.pure real_type{M <: Mesh}(::Type{M}) = M.parameters[3]
real_type{M <: Mesh}(::M) = real_type(M)

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

function connectivity{C1 <: Cell, C2 <: Cell}(msh, ::C1, ::C2)
  assert(subcell(C1, dim_t(C2)) == C2)
  connectivity(msh, dim_t(C1), dim_t(C2))[C1]
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
function geometry{M <: Mesh}(mesh::M)
  mapreduce(chain, decompose(connectivity(mesh, Codim{0}(), Dim{0}()))) do conns
    map(graph(conns)) do idx, conn
      Geometry{cell_type(idx), world_dim(M), real_type(M)}(map(vidx -> coordinates(mesh)[convert(Int, vidx)], conn))
    end
  end
end

cells(mesh) = connectivity(mesh, Codim{0}(), Dim{0}())

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
