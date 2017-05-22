using TipiFEM.Meshes

using Base.Test

info("Testing Dim, Codim")

info("Testing generic mesh interface")

module TestMesh
  using StaticArrays
  using Base.Test

  using TipiFEM.Meshes
  using TipiFEM.Simple1DMesh

  @test facet(Edge) == Vertex
  @test subcell(Edge, Codim{1}()) == Vertex
  @test skeleton(Edge) == (Vertex,Edge)

  @test face_count(Edge, Vertex) == 2
  @test vertex_count(Edge) == 2

  # todo: fix
  #@inferred Mesh(Edge)

  edge_conn = Connectivity{Edge, Vertex}(1, 2)
  @test vertex(edge_conn, 1) == Index{Vertex}(1)
  @test vertex(edge_conn, 2) == Index{Vertex}(2)

  edge_geo = Geometry{Edge, dim(Edge), Float64}(
    SVector{1, Float64}(1.),
    SVector{1, Float64}(3.)
  )
  @test volume(edge_geo) == 2.

  mesh = Mesh(Edge)
  x = [0, 1]
  N = 1e6
  h = (x[2]-x[1])/(N+1)
  #@time for xi in x[1]+h:h:x[2]-h
  #  add_vertex!(mesh, xi)
  #end
  #@time for (idx1, idx2) in zip(vertices(mesh)[1:end-1], vertices(mesh)[2:end])
  #  add_cell!(mesh, Edge, idx1, idx2)
  #end
end


using TipiFEM.PolytopalMesh

################################################################################
# cell types
################################################################################
@test Polytope"3-node triangle"() == Polytope("3-node triangle")
@test vertex_count(Polytope"3-node triangle") == 3
#@test vertex_count(Polytope"6-node second order triangle") == 6

################################################################################
# cell index
################################################################################
# logic operations
@test 6 < Index"3-node triangle"(7) < 8
@test 8 > Index"3-node triangle"(7) > 6
@test 7 <= Index"3-node triangle"(7) <= 7
@test 7 >= Index"3-node triangle"(7) >= 7
let i1=Index"3-node triangle"(2), i2=Index"3-node triangle"(3)
  @test i1 == i1
  @test i1 <= i2
  @test i1 <  i2
  @test i2 >  i1
  @test i2 >= i1
end
@test Index"3-node triangle"(7) < Index"3-node triangle"(8)
@test Index"3-node triangle"(8) > Index"3-node triangle"(7)
@test Index"3-node triangle"(7) <= Index"3-node triangle"(8)
@test Index"3-node triangle"(8) >= Index"3-node triangle"(7)
@test Index"3-node triangle"(7) <= Index"3-node triangle"(7)
@test Index"3-node triangle"(7) >= Index"3-node triangle"(7)

# arithmetic
@test Index"3-node triangle"(7)+1 == Index"3-node triangle"(8)
@test Index"3-node triangle"(7)-1 == Index"3-node triangle"(6)

################################################################################
# mesh function
################################################################################
info("Testing mesh function")
@inferred MeshFunction(Polytope"1-node point", Int)
info(" - HomogenousMeshFunction")
let mf=MeshFunction(Polytope"1-node point", Int)
  push!(mf, 1)
  push!(mf, 2.)
  @test length(mf) == 2
  @test mf[1] == 1
  @test mf[2] == 2
  @test mf[first(domain(mf))] == 1
  @test mf[last(domain(mf))] == 2
  @inferred mf[1]
  @inferred mf[first(domain(mf))]
  @inferred first(domain(mf))
  @inferred first(image(mf))
  @inferred eltype(mf)
  @inferred idxtype(mf)
  empty!(mf)
  @test length(mf) == 0
end
let indices = Index"3-node triangle"(1):Index"3-node triangle"(10),
    values = 1:10
  @inferred MeshFunction(indices, values)
  mf = MeshFunction(indices, values)
  @test length(mf) == 10
end
info(" - HetereogenousMeshFunction")
let mf=MeshFunction(Union{Polytope"3-node triangle", Polytope"4-node quadrangle"}, Int)
  push!(mf, Polytope"3-node triangle", 1)
  push!(mf, Polytope"4-node quadrangle", 2.)
  @test mf[Index"3-node triangle"(1)]==1
  @test mf[Index"4-node quadrangle"(1)]==2
  @test length(mf[Polytope"3-node triangle"]) == 1
  @test length(mf[Polytope"4-node quadrangle"]) == 1
  @test mf[Polytope"3-node triangle"][1] == 1
  @test mf[Polytope"4-node quadrangle"][1] == 2
  @test length(decompose(mf)) == 2
  @test length(mf) == 2
  @inferred mf[Polytope"3-node triangle"]
  @inferred first(mf[Polytope"3-node triangle"])
  map(decompose(mf)) do mf2
    @inferred first(mf2)
  end
  mf[Polytope"3-node triangle"] = MeshFunction(Polytope"3-node triangle", Int)
  @test length(mf) == 1
  empty!(mf)
  @test length(mf) == 0
end

################################################################################
# cell connectivity
################################################################################
#c1 = Connectivity"3-node triangle → 1-node point"(1, 2, 3)
info("Testing mesh interface")
info(" - homogenous mesh with triangular elements")
let mesh = Mesh(Polytope"3-node triangle")
  # add some vertices
  add_vertex!(mesh, 0, 0)
  add_vertex!(mesh, 0, 1)
  add_vertex!(mesh, 1, 1)
  @test coordinates(mesh)[Index"1-node point"(1)]==[0, 0]
  @test coordinates(mesh)[Index"1-node point"(2)]==[0, 1]
  @test coordinates(mesh)[Index"1-node point"(3)]==[1, 1]
  @test number_of_cells(mesh, Polytope"1-node point") == 3

  # ensure that no cells with index zero may be added
  @test_throws ErrorException add_cell!(mesh, Polytope"3-node triangle", 0, 2, 3)

  # connect vertices
  add_cell!(mesh, Polytope"3-node triangle", 1, 2, 3)
  @test length(geometry(mesh)) == 1
  @test geometry(mesh)[Index"3-node triangle"(1)]==Geometry{Polytope"3-node triangle", 2, Float64}(
    (0, 0),
    (0, 1),
    (1, 1))
  @test number_of_cells(mesh, Polytope"3-node triangle") == 1

  # populate connecitivity
  populate_connectivity!(mesh)
end
info(" - homogenous mesh with quadrilateral elements")
# v3 ---- v2 ---- v5
#  |       |       |
#  |       |       |
# v0 ---- v1 ---- v4
let mesh = Mesh(Polytope"4-node quadrangle")
  # add some vertices
  add_vertex!(mesh, 0, 0)
  add_vertex!(mesh, 0, 1)
  add_vertex!(mesh, 1, 1)
  add_vertex!(mesh, 1, 0)
  add_vertex!(mesh, 2, 0)
  add_vertex!(mesh, 2, 1)
  @test number_of_cells(mesh, Polytope"1-node point") == 6

  # connect vertices
  add_cell!(mesh, Polytope"4-node quadrangle", 1, 2, 3, 4)
  add_cell!(mesh, Polytope"4-node quadrangle", 2, 5, 6, 3)
  @test number_of_cells(mesh, Polytope"4-node quadrangle") == 2

  # populate connecitvity
  populate_connectivity!(mesh)

  @test number_of_cells(mesh, Polytope"2-node line") == 7
  # the mesh should contain 7 edges
  @test length(topology(mesh)[Polytope"2-node line"]) == 7

  # populate again and check that the number of edges is still 7
  populate_connectivity!(mesh)
  @test length(topology(mesh)[Polytope"2-node line"]) == 7
end
#info(" - heterongenous mesh with triangular and quadrilateral elements")
#let mesh = Mesh(Union{Polytope"3-node triangle", Polytope"4-node quadrangle"})
#  # add some vertices
#  add_vertex!(mesh, 0, 0)
#  add_vertex!(mesh, 0, 1)
#  add_vertex!(mesh, 1, 1)
#  add_vertex!(mesh, 1, 0)
#  add_vertex!(mesh, 3, 0.5)
#  @test number_of_cells(mesh, Polytope"1-node point") == 5
#
#  # connect vertices
#  add_cell!(mesh, Polytope"4-node quadrangle", 1, 2, 3, 4)
#  add_cell!(mesh, Polytope"3-node triangle", 2, 5, 3)
#  @test number_of_cells(mesh, Polytope"4-node quadrangle") == 1
#  @test number_of_cells(mesh, Polytope"3-node triangle") == 1
#
#  # populate connecitvity
#  populate_connectivity!(mesh)
#
#  @test number_of_cells(mesh, Polytope"2-node line") == 6
#end

################################################################################
# cell geometry
################################################################################
using StaticArrays
info("Testing cell geometry")
const coord_t = SVector{2, Float64}

let ref_tria = reference_element(Polytope"3-node triangle"),
    tria = Geometry{Polytope"3-node triangle", 2, Float64}(
             coord_t(0, 0),
             coord_t(2, 0),
             coord_t(0, 2))
  @test volume(ref_tria) ≈ 1/2
  @test volume(tria) ≈ 2
  @test det(jacobian_transposed(ref_tria, coord_t(0, 0))) ≈ 1
  @test det(jacobian_transposed(tria, coord_t(0, 0))) ≈ 4
  @test local_to_global(tria, point(ref_tria, 1)) == coord_t(0, 0)
  @test local_to_global(tria, point(ref_tria, 2)) == coord_t(2, 0)
  @test local_to_global(tria, point(ref_tria, 3)) == coord_t(0, 2)
end

#
# type stability
#
info("Testing type stability")
let mesh = Mesh(Polytope"4-node quadrangle")
  # add some vertices
  add_vertex!(mesh, 0, 0)
  add_vertex!(mesh, 0, 1)
  add_vertex!(mesh, 1, 1)
  add_vertex!(mesh, 1, 0)
  add_vertex!(mesh, 2, 0)
  add_vertex!(mesh, 2, 1)

  # connect vertices
  add_cell!(mesh, Polytope"4-node quadrangle", 1, 2, 3, 4)
  add_cell!(mesh, Polytope"4-node quadrangle", 2, 5, 6, 3)

  # populate connecitvity
  populate_connectivity!(mesh)

  #
  # test type stability
  #
  @inferred number_of_cells(mesh, Polytope"2-node line")
  @inferred connectivity(mesh, Codim{0}(), Dim{0}())
  @inferred connectivity(mesh, Codim{0}(), Codim{0}())
end

#
# Shape functions
#


exit()

#
# unfinished, slow stuff below
#
module TestMesh

  # test CellRef type
  #let msh = Mesh(Edge),
  #    idx = Index{Vertex}(1),
  #    ref = CellRef(msh, idx)

  #  # the inference algorithm of a standard julia installation has a
  #  #  MAX_TYPE_DEPTH value of 7 which is not high enough to allow
  #  #  inference of code that contains the mesh type. We can change
  #  #  this by compiling julia with a higher value (e.g. 20).
  #  #  To do so just change the value of MAX_TYPE_DEPTH in base/inference.jl
  #  @test Base.Core.Inference.type_too_complex(typeof(msh), 0) == false

  #  @inferred CellRef(msh, idx)

  #  @test mesh(ref) == msh
  #  @test index(ref) == idx
  #  @inferred mesh(ref)
  #  @inferred index(ref)
  #end

  # test mesh
  let
    msh = Mesh(Edge)

    @test cell_type(msh) == Edge
    @test cell_type(msh, 0) == Vertex
    println(vertex_type(cell_type(msh)))

    add_vertex!(msh, 0)
    add_vertex!(msh, 1)
    add_cell!(msh, Edge, 1, 2)

    vtx = CellRef(msh, Index{Vertex}(2))
    cell = CellRef(msh, Index{Edge}(1))

    @inferred coordinates(vtx)

    #@test_throws AssertionError coordinates(msh, Index{Edge}(2))
    @test coordinates(vtx) == coordinates(msh, Index{Vertex}(2)) == SVector{1, Float64}(1)
    @test coordinates(msh, Index{Vertex}(1)) != coordinates(msh, Index{Vertex}(2))
    @test volume(cell) == 1.

    display(@code_warntype coordinates(vertex(cell, 2)))
    display(@code_warntype volume(cell))
  end

  exit()

  v1 = Vertex(0)
  v2 = Vertex(1)
  e = Edge(v1, v2)

  @test volume(e) == 1.
  @test skeleton(Edge) == (Vertex,)

  # create a mesh
  mesh = Mesh(Edge)

  # check that the return type of the connectivity arrays is inferred correctly
  @inferred mesh.topology[Dim{1}(), Dim{0}()]
  @inferred connectivity(mesh, Dim{1}(), Dim{0}())

  x = [0, 1]
  N = 10
  h = (x[2]-x[1])/(N+1)
  @time for xi in x[1]+h:h:x[2]-h
    add_vertex!(mesh, xi)
  end

  @time for (idx1, idx2) in zip(domain(vertices(mesh))[1:end-1], domain(vertices(mesh))[2:end])
    add_cell!(mesh, Edge, idx1, idx2)
  end

  A = zeros(N-2, N-2)
  for j in 1:size(A, 2), i in 1:size(A, 1)
    println(i, " ", j)
  end

  exit()

  #println(mesh.topology[Dim{1}(), Dim{0}()])
  #println("code: ", @code_warntype mesh.topology[Dim{1}(), Dim{0}()])
  #exit()
  #println("code: ", @code_warntype connectivity(mesh, Dim{1}(), Dim{0}()))
  #println(@code_warntype add_cell!(mesh, Edge, 0, 1))
  Profile.clear_malloc_data()
  @time for i=1:Int(1e6)
    add_cell!(mesh, Edge, i, i+1)
  end
  #println(mesh)
  #display(graph(Meshes.connectivity(mesh, 1, 0)))

  #for cell, cell_conn in connectivity(mesh, Codim{0}, Dim{0})
#    for local_idx_i=1:2
#      for local_idx_j=1:2
#        global_idx_i = dof(conn, basis, local_idx_i)
#        global_idx_j = dof(conn, basis, local_idx_j)
#        galerkin_matrix[global_idx_i, global_idx_j] = a()
#      end
#    end
#  end


  end

  u(x) = sin(x/π)
  f(x) = -sin(x/π)
  h=1/8
  x = [0, 1]
  y = u.(x)
  #local_basis = (0 < x < 0.5) ? 2x : 2(1-x)
  function global_basis(x0, h)
    x -> begin
      if 0. <= x <= h/2
        2(x-x0)/h
      elseif h/2 < x <= h
        -2(x-x0)/h+2
      else
        0
      end
    end
  end
  #println(global_basis(0, 1).(0:0.1:2))
  integrate(f, x0, x1, N=10) = 1/N * sum(f.(x0:((x1-x0)/(N-1)):x1))

  # assemble gallerkin matrix
  A = zeros(length(enumerate(x[1]:h:x[2])), length(enumerate(x[1]:h:x[2])))
  for (i, xi)=enumerate(x[1]:h:x[2]-h)
    #println("interval: ", xi, " ", xi+h)
    for (j, xj)=enumerate(x[1]:h:x[2]-h)
      #println(" interval: ", xj, " ", xj+h)
      A[i, j] = integrate(x -> global_basis(xi, h)(x)*global_basis(xj, h)(x), x[1], x[2])
    end
  end
  #display(A)
end

exit()

immutable Vertex <: Cell end
dim(::Type{vertex}) = 0

################################################################################
# cell types
################################################################################
@test Polytope"3-node triangle"() == Polytope("3-node triangle")
@test vertex_count(Polytope"3-node triangle") == 3
@test vertex_count(Polytope"6-node second order triangle") == 6

################################################################################
# cell index
################################################################################
# logic operations
@test 6 < Index"3-node triangle"(7) < 8
@test 8 > Index"3-node triangle"(7) > 6
@test 7 <= Index"3-node triangle"(7) <= 7
@test 7 >= Index"3-node triangle"(7) >= 7
# arithmetic
@test Index"3-node triangle"(7)+1 == Index"3-node triangle"(8)
@test Index"3-node triangle"(7)-1 == Index"3-node triangle"(6)

################################################################################
# cell connectivity
################################################################################
c1 = Connectivity"3-node triangle → 1-node point"(1, 2, 3)
Polytope"3-node triangle"(0, 1, 2)

mesh = Mesh(world_dim=2, real_type=Float64)

add_vertex!(mesh, 0, 0)
add_vertex!(mesh, 0, 1)
add_vertex!(mesh, 1, 1)

add_cell!(mesh, Polytope"3-node triangle", 0, 1, 2)

println(mesh)
