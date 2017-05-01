using Base.return_types

add_cell_initializer((K) -> raw_topology_tuple_type(K))

const topology_tuple_type_cache = Dict{Type, DataType}()

using Base.uniontypes

@dim_dispatch @Base.pure function connectivity_types(K, i::Dim, j::Dim)
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
      topology_matrix[i+1, j+1]=connectivity_types(K, Dim{i}(), Dim{j}())
    end
  end
  topology_matrix
end

@Base.pure function raw_topology_tuple_type(K)
  mesh_dim = dim(K)
  ts = []
  let topology_matrix = connectivity_types(K)
    for i in 0:mesh_dim
      for j in 0:mesh_dim
        push!(ts, first(return_types(MeshFunction, (Type{subcell(K, i)},
                                     Type{topology_matrix[i+1, j+1]}))))
       end
     end
   end

  T = Tuple{ts...}
  topology_tuple_type_cache[K] = T # cache the result
  T
end

@Base.pure function topology_tuple_type(K)
  Tuple{(Base.RefValue{Nullable{T}} for T in raw_topology_tuple_type(K).parameters)...}
end

@computed type MeshTopology{K}
  topology::topology_tuple_type(K)

  function MeshTopology()
    new{K}(((Base.RefValue{Nullable{T}}(Nullable{T}()) for T in raw_topology_tuple_type(K).parameters)...))
  end
end

@dim_dispatch function initialize_connectivity!{K}(topology::MeshTopology{K}, i::Dim, j::Dim)
  topology[i, j] = MeshFunction(subcell(K, i), connectivity_types(K)[convert(Int, i)+1, convert(Int, j)+1])
end

using SimpleRepeatIterator

"""
Initialize mesh connecitivity
"""
function populate_connectivity!(msh)
  # initialize connectivity
  initialize_connectivity!(topology(msh), Codim{0}(), Codim{1}())
  initialize_connectivity!(topology(msh), Codim{0}(), Codim{0}())
  initialize_connectivity!(topology(msh), Codim{1}(), Dim{0}())
  # boundary edges
  boundary_edge_count = 0
  # populate connectivity
  map(decompose(connectivity(msh, Codim{0}(), Dim{0}()))) do el_conn
    let K = cell_type(el_conn)
      # retrieve all edges from the facets
      # - every edge occurs twice, but with a different parent triangle
      #   which is encoded in the index of the flattened mesh function
      # - the edges have no identity
      ul_edges = flatten(map(facets, el_conn))
      # store local indices
      ul_edges_lidx = MeshFunction(domain(ul_edges), repeat(1:vertex_count(K), outer=length(el_conn)))
      # allocate an array for the edges (incidence relation codim 1 -> dim 0)
      edges = connectivity(msh, Codim{1}(), Dim{0}())
      # allocate an array for the incidence relation codim 0 -> codim 1
      el_facets = MeshFunction(K, Connectivity{K, subcell(K)})
      resize!(el_facets, length(el_conn), ((0 for i in 1:vertex_count(K))...))
      # allocate an array for the neighbourhood relation
      neighbours = MeshFunction(K, NeighbourConnectivity{K})
      resize!(neighbours, length(el_conn), ((0 for i in 1:vertex_count(K))...))
      # sort unlabeled edges by vertex list
      A = sort(zip(ul_edges_lidx, ul_edges), by=(x)->x[2])
      # sort unlabeled edges by vertex list reversed
      B = sort(zip(ul_edges_lidx, ul_edges), by=x->reverse(x[2]))

      #println("A ", map(x->(x[1], (convert(Int, x[2][1]), convert(Int, x[2][2]))),image(A)))
      #println("B ", map(x->(x[1], (convert(Int, x[2][1]), convert(Int, x[2][2]))),image(B)))
      #println()

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
          #println(a, " ", b, " ")
          #println("A ", map(x->(x[1], (convert(Int, x[2][1]), convert(Int, x[2][2]))),image(A)[i_a:end]))
          #println("B ", map(x->(x[1], (convert(Int, x[2][1]), convert(Int, x[2][2]))),image(B)[i_b:end]))
          # higher i_b until we find either find the opposing edge or i_b points
          #  to an edge whose end is not equal to the source of `a` anymore.
          # note that all edges which are discarded by highering i_b are
          #  boundary edges
          while a[1] == b[2] && a[2] >= b[1]
            #@show a[1]
            #@show b[2]
            #@show a[1]<=b[2]
            #println()
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
          #println("A ", map(x->(x[1], (convert(Int, x[2][1]), convert(Int, x[2][2]))),image(A)[i_a:end]))
          #println("B ", map(x->(x[1], (convert(Int, x[2][1]), convert(Int, x[2][2]))),image(B)[i_b:end]))
          #println(is_on_boundary)
          #println()
          # if the current edge is a boundary edge we just add it to the edge array
          # if the edge is not on the boundary we only add it if the index
          #  of its source is higher then the index of its sink. This ensures
          #  the we add every edge only once.
          if is_on_boundary || a[1]<a[2]
            push!(edges, a)
            el_facets[i_a, il_a] = last(domain(edges))
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
      topology(msh)[Codim{0}(), Codim{1}()][K] = el_facets
      topology(msh)[Codim{0}(), Codim{0}()][K] = neighbours
    end
  end
  # integrity check
  let edges=connectivity(msh, Codim{1}(), Dim{0}())
    assert(mapreduce(K -> facet_count(K)*number_of_cells(msh, K),
                     +, cell_types(msh))+boundary_edge_count == 2*length(edges))
  end
end

"retrieve index at which the connectivity i → j is stored"
@Base.pure conn_index{K <: Cell}(::Type{K}, i::Int, j::Int) = i*(dim(K)+1)+j+1 # row major
#conn_index{K <: Cell}(::Type{K}, i::Int, j::Int) = (dim(K)-i)*(dim(K)+1)+(dim(K)-j)+1

import Base: setindex!, getindex

@dim_dispatch function setindex!{K <: Cell, i, j}(topology::MeshTopology{K}, mf::MeshFunction, ::Dim{i}, ::Dim{j})
  topology.topology[conn_index(K, i, j)][] = mf
end

@dim_dispatch function hasindex{K <: Cell, i, j}(topology::MeshTopology{K}, ::Dim{i}, ::Dim{j})
  !isnull(topology.topology[conn_index(K, i, j)][])
end

@dim_dispatch @generated function getindex{K <: Cell, i, j}(topology::MeshTopology{K}, ::Dim{i}, ::Dim{j})
  :(get(topology.topology[$(conn_index(K, i, j))][]))
end

"vertex connectivity of all `C` cells"
function getindex{K <: Cell, C <: Cell}(topology::MeshTopology{K}, ::Union{C, Type{C}})
  topology[dim_t(C), Dim{0}()][C]
end

@prototyping_only function getindex{T <: MeshTopology}(topology::T, i::Int, j::Int)
  topology[Dim{i}(), Dim{j}()]
end
