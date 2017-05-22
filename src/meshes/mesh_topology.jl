using Base.return_types

add_cell_initializer((K) -> topology_tuple_type(K, raw=true))

const topology_tuple_type_cache = Dict{Type, DataType}()

using Base.uniontypes

@dim_dispatch @Base.pure function connectivity_type(K, i::Dim, j::Dim)
  Cj=subcell(K, j)
  Cis=uniontypes(subcell(K, i))
  Union{map(Cis) do Ci
    if i==Dim{0}() && j == Dim{0}()
      Void
    elseif i<j
      VariableConnectivity{Cj} # subcell(K, i),
    elseif i==j
      fulltype(NeighbourConnectivity{Ci})
    else
      fulltype(Connectivity{Ci, Cj})
    end
  end...}
end

@Base.pure function connectivity_types(K)
  mesh_dim = dim(K)
  topology_matrix = Array{Any, 2}(mesh_dim+1, mesh_dim+1)
  c = 1
  for i in 0:mesh_dim
    for j in 0:mesh_dim
      @assert c == conn_index(K, i, j); c+=1
      topology_matrix[i+1, j+1]=connectivity_type(K, Dim{i}(), Dim{j}())
    end
  end
  topology_matrix
end

@Base.pure function topology_tuple_type(K; raw=false)
  mesh_dim = dim(K)
  ts = []
  let topology_matrix = connectivity_types(K)
    for i in 0:mesh_dim
      for j in 0:mesh_dim
        T = first(return_types(MeshFunction, (Type{subcell(K, i)},
                                     Type{topology_matrix[i+1, j+1]})))
        isa(T, DataType) ||
          error("Return type of MeshFunction constructor could not be inferred")

        push!(ts, T)
       end
     end
   end

  T = Tuple{ts...}
  topology_tuple_type_cache[K] = T # cache the result
  T
end

"""
Two dimensional array like datastructure storing all mesh connectivities
"""
@computed immutable MeshTopology{K}
  data::topology_tuple_type(K)
  populated::Array{Bool, 2}
end

function MeshTopology(::Type{K}) where K <: Cell
  data = MeshFunction[]
  for i in 0:dim(K)
    for j in 0:dim(K)
      push!(data, MeshFunction(subcell(K, Dim{i}()),
                               connectivity_type(K, Dim{i}(), Dim{j}())))
    end
  end
  MeshTopology{K}((data...), zeros(Bool, dim(K)+1, dim(K)+1))
end

"""
Mark mesh connectivity i → j as populated
"""
@dim_dispatch function mark_populated!{K, i, j}(topology::MeshTopology{K}, ::Dim{i}, ::Dim{j})
  topology.populated[convert(Int, i)+1, convert(Int, j)+1]=true
end

"""
Clear mesh connecitvity i → j. Put differently remove all cell connectivity from
the mesh connectivity i → j.
"""
@dim_dispatch function clear_connectivity!{K}(topology::MeshTopology{K}, i::Dim, j::Dim)
  ispopulated(topology, i, j) && empty!(topology[i, j])
  topology.populated[convert(Int, i)+1, convert(Int, j)+1]=false
  nothing
end

"retrieve index at which the connectivity i → j is stored"
@Base.pure conn_index{K <: Cell}(::Type{K}, i::Int, j::Int) = i*(dim(K)+1)+j+1 # row major
#conn_index{K <: Cell}(::Type{K}, i::Int, j::Int) = (dim(K)-i)*(dim(K)+1)+(dim(K)-j)+1

import Base: setindex!, getindex

#"set mesh connecitvity i → j"
#@dim_dispatch function setindex!{K <: Cell, i, j}(topology::MeshTopology{K}, mf::MeshFunction, ::Dim{i}, ::Dim{j})
#  topology.data[conn_index(K, i, j)][] = mf
#end
"is the mesh connectivity i → j populated"
@dim_dispatch @inline function ispopulated{K <: Cell, i, j}(topology::MeshTopology{K},
    ::Dim{i}, ::Dim{j})
  topology.populated[convert(Int, i)+1, convert(Int, j)+1]
end

"retrieve mesh connectivity i → j"
@dim_dispatch @inline function getindex{K <: Cell, i, j}(topology::MeshTopology{K},
    ::Dim{i}, ::Dim{j}, allow_unpopulated=false)
  @boundscheck allow_unpopulated || ispopulated(topology, Dim{i}(), Dim{j}()) ||
    throw(BoundsError(topology, (Dim{i}(), Dim{j}())))
  topology.data[conn_index(K, i, j)]
end

"mesh vertex connectivity of all `C` cells"
function getindex{K <: Cell, C <: Cell}(topology::MeshTopology{K}, ::Union{C, Type{C}})
  topology[dim_t(C), Dim{0}()][C]
end

"retrieve mesh connectivity i → j"
@prototyping_only function getindex{T <: MeshTopology}(topology::T, i::Int, j::Int)
  topology[Dim{i}(), Dim{j}()]
end
