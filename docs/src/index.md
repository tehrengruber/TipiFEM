# TipiFEM - A framework for implementation of finite element methods

This is the documentation to TipiFEM, a framework for the implementation of finite element methods in julia.
The purpose of this documentation is to give the required mathematical background on which TipiFEM builds
on, explain the software design of TipiFEM and how it may be used.

TipiFEM is still in an early stage. Currently only the implementation of
low order piecewise polynomial finite element Galerkin discretizations
has been completed.

The framework is separated into different modules

- `TipiFEM.FE`: Code for the implementation of finite element methods
- `TipiFEM.Meshes`: Generic code for the implementation of Meshes
- `TipiFEM.Quadrature`: Generic code for the approximate solution of integrals on mesh cells
- `TipiFEM.PolytopalMesh`: (Incomplete) implementation of a mesh with polytopal cells.

## Mesh

We define a mesh $\cal M$ as subdivision of a bounded set $\Omega \subset \mathbb{R}^d$ into
open cells of dimension $< d$, where a cell is in general a topological space homeomorph
to an open ball. The mesh is then used to represent the domain on which the problem
we are trying to solve with our FEM solver is posed on.

The mesh dimension or simply dimension of a mesh is the largest dimension of one of its cells.
The dimension of the ambient space $d$ of the mesh is called the world dimension. We will
generally use the notation $K$ or $K_i$ to talk about cells of codimension zero.

*High level view*

A mesh in TipiFEM is stored as an object of type `Mesh` consisting of multiple
`MeshFunction`s, which store the vertex coordinates, cell topology and additional cell attributes.

*Restrictions*

In TipiFEM we further restrict $\Omega$ to be a manifold surface and
the cells to be polytopes or entities with the same combinatorial structure as polytopes
(e.g. curvilinear elements). The first restriction is due to the algorithm that is used
to wire up the incidence relation between different cells and might be lifted by a different
implementation there. The mesh data structure itself may easily store non-manifold meshes.
The restriction on the type of cells is only made since I wasn't aware of any problem
that may be solved using finite elements where the cells have a different combinatorial
structure then polytopes.

*Features*

- Hybrid mesh with triangular and quadrilateral cells
- Quadrature rules for triangular or quadrilateral cells with order up to 25 respectively 36
- Generic degree of freedom mapper for continuous Lagrangian finite element methods
- Generic finite element matrix and vector assemblers

Now that we have defined what a mesh is we proceed with details of its implementation. A
minimal representation of a polytopal-like mesh consists of a set of vertices
${\cal V}$, where each vertex has a unique identifier and coordinate $x \in \Omega$,
and a set of cells ${\cal K}$, where each cell has again a unique identifier and
set of vertices.

```@docs
TipiFEM.Meshes.Mesh
TipiFEM.Meshes.world_dim
TipiFEM.Meshes.cell_type
TipiFEM.Meshes.mesh_dim
TipiFEM.Meshes.real_type
```

### MeshInterface

immutable Edge <: Cell end
dim(::Type{Edge}) = 1
face_count(::Type{Edge}, ::Type{Vertex}) = 2
facet(::Type{Edge}) = Vertex
volume(geo::Geometry{Edge}) = point(geo, 2)[1] - point(geo, 1)[1]

Many aspects of a mesh implementation are
The topology and geometry of the mesh
TipiFEM contains a generic mesh implementation that may be used for various

__Simple1DMesh__

__PolytopalMesh__

### MeshFunction

The basic building block to store both geometrical and combinatorial information
of a mesh is what we call a mesh function, which is just a discrete
function that maps cell-indices of cells with a fixed dimension to arbitrary objects.

```@docs
TipiFEM.Meshes.AbstractMeshFunction{II, VI}
```

Lets start with some examples of how mesh functions may be used to store the
topology of a mesh. We begin with a simple unstructed two dimensional mesh as of
Fig #X by constructing a mesh function that maps indices of triangles to vertex indices.

```@repl meshfunction_ex_1
using TipiFEM, TipiFEM.Meshes, TipiFEM.PolytopalMesh
mf = MeshFunction(Polytope"3-node triangle", NTuple{3, Index"1-node point"})
push!(mf, (1, 4, 2))
push!(mf, (1, 3, 4))
push!(mf, (1, 3, 4))
```

The value table `mf` is then:

| i                         | f(i)                                                                     |
| ------------------------- | ------------------------------------------------------------------------ |
| Index"3-node triangle"(1) | (Index"1-node point"(1), Index"1-node point"(4), Index"1-node point"(2)) |
| Index"3-node triangle"(2) | (Index"1-node point"(1), Index"1-node point"(3), Index"1-node point"(4)) |

Note how automatically two indices for the triangles are created and how the tuple of integers
is converted into a tuple of indices. We can now look at the internal storage representation
of the mesh function.

```@repl meshfunction_ex_1
domain(mf)
image(mf)
```

We see that the indices in this case are stored as a leightweight `UnitRange` object of triangle indices, where
a UnitRange object stores only the starting and end value of a range eliminated the need
to store all indices or to store indices implicitly by their index. The image of the mesh function
however is stored as a regular Array of tuples.

Now that we have a mesh function that contains some values we can further explore
some aspects of the `MeshFunction` interface. The mesh function may be evaluated
at any index by calling the `[]` operator.
```@repl meshfunction_ex_1
mf[Index"3-node triangle"(1)]
```
In case we know the integer index of the value inside the values array we can also evaluate
the function using that index instead of the triangle index, which is not an integer by itself,
eliminating the computation of the integer index from the triangle index.
```@repl meshfunction_ex_1
mf[1]
```
Note that in this case the integer index and the triangle index are equal, which
will not be the case if the domain is not a UnitRange beginning at one (see example #X).

Iteration over all values works just like with a regular Array:
```@repl meshfunction_ex_1
for connectivity in mf
  println(connectivity)
end
```
However in some cases the indices are also required for which the `graph` function is
useful:
```@repl meshfunction_ex_1
for (index, connectivity) in graph(mf)
  println("index: ", index, ", connectivity: ", connectivity)
end
```
Just like iteration over a mesh function works as if it was a regular array,
many other functions that work on regular arrays also work with mesh functions
(e.g. `map`, `reduce` etc.). We might for example want obtain all edges of the triangles
stored in `mf`. Assuming a fixed node numbering convention, where the vertices
of each triangle are stored in counter clockwise order, we can get all edges like this:
```@repl meshfunction_ex_1
# construct a mesh function that maps triangles to the connectivity of all of its edges
triangle_edges = map(mf) do connectivity
  ((connectivity[1], connectivity[2]),
   (connectivity[2], connectivity[3]),
   (connectivity[3], connectivity[1]))
end
# remove duplicate edges
edges = MeshFunction(Edge, NTuple{2, Index"1-node point"})
for edges in triangle_edges
  for edge in triangle_edges
    if edge[2]<edge[1]
      push!(edges, edge)
    end
  end
end
```

Now we proceed with a more complex example

map indices of triangles and quadrangles to vertex
indices. For this purpose we can either construct two mesh functions, one for the
triangles and one for the quadrangles, and union them or create a single mesh
function that stores both triangles and quadrangles.

```
mf1 = MeshFunction(Polytope"3-node triangle", Array{Vertex, 1})()
mf2 = MeshFunction(Polytope"3-node quadrangle", Array{Vertex, 1})()
push!(mf1, [1, 4, 2])
push!(mf2, [1, 3, 4])
mf = mf1 ∪ mf2
```

The construction of a mesh cell is straightforward by calling

The mesh is represented as a set of mesh functions, which we define as a discrete
function that maps cell-indices of cells with a fixed dimension to arbitrary objects.

The central data-type used by the mesh is a `MeshFunction`. It is defined as a discrete
function that maps cell-indices of cells with a fixed dimension to arbitrary objects.
`MeshFunctions` are used to store the connectivity, material parameters
or user defined attributes of mesh cells. Together a set of mesh functions
may then define all properties of the cells in the mesh and give them an identity
through their index. Currently there exist two different concrete types of MeshFunctions.

The discreteness of a MeshFunction is expressed by

Essentially a MeshFunction is just an interface around a pair of equally sized
`AbstractArrays` storing cell indices and values. Since the meaning of a MeshFunction
is only clear in the context it is being used the

__Scalar discrete function__

#### A simple mesh

__Connectivity__

```
trias = MeshFunction(Polytope"3-node triangle", SVector{3, Index"1-node point"})
push!(trias, [v1, v4, v2])
push!(trias, [v1, v3, v4])
```

__Geometry__

```
nodes = MeshFunction(Polytope"1-node point", SVector{2, Float64})
v1 = push!(nodes, [0, 0])
v2 = push!(nodes, [0, 1])
v3 = push!(nodes, [1, 0])
v4 = push!(nodes, [0.5, 0.5])
```

In combination with the connectivity the vertex coordinates may be used to calculate
the area of all triangles

```@docs
TipiFEM.Meshes.domain
TipiFEM.Meshes.image
TipiFEM.Meshes.graph
TipiFEM.Meshes.idxtype
TipiFEM.Meshes.cell_type
TipiFEM.Meshes.eltype
TipiFEM.Meshes.length
TipiFEM.Meshes.start
TipiFEM.Meshes.done
TipiFEM.Meshes.next
TipiFEM.Meshes.getindex
TipiFEM.Meshes.map
```

The MeshFunction type has two fields, one for the cell indices

The actual storage layout

__Example__
Let's begin with an discrete scalar function on a set of triangles. Such a function
may for example assign each triangle its volume.
```
# construct a mesh function mapping triangles to Float64 (Real) values
mf=MeshFunction(Polytope"3-node triangle", Float64)
```
Adding values is as easy as:
```
push!(mf, Index"3-node triangle"(1), 1)
push!(mf, Index"3-node triangle"(1), 2)
```
Now just like with a regular array `sum(mf)`
```
sum(mf)
```


```
MeshFunction(Union{Polytope"4-node quadrangle", Polytope"3-node triangle"}, Float64)
```

**MeshTopology**

The topology of the mesh is represented by incidence relations stored as a
$(d+1) \times (d+1)$ Matrix 𝕀 of `Connectivity`-valued `MeshFunction`s, where
𝕀$(d_1, d_2)$ stores the incident $d_2$ dimensional cells of $d_1$ dimensional cells.

### MeshCell

Since the cells of a mesh are stored implicitly through `MeshFunctions` their data-types
also do not store any information. A concrete mesh `Cell` (e.g. a `Polytope`) is therefore
implemented as a singleton type in julia, whose only purpose is to allow distinguishing
different types of cells. Intrinsic properties of mesh cells may then be defined by simple functions taking a concrete `MeshCell` as an argument. The dimension of a 3-node triangle for example is defined like this:

```
dim(::Polytope"3-node triangle") = 2
```

#### Connectivity


#### Geometry



#### Misc

**Dim, Codim**

The `Dim` and `Codim` types are used to represent dimension and codimension information.
Since in most situations the dimension and the codimension is already known at compile time
these types are implemented as singleton types that store the value of the (co)dimension
as a parametric type (note that the value is an integer and not a type). This allows us
to dispatch on the (co)dimension, which is especially handy in cases, where the knowledge
of the (co)dimension at compile time allows generation of efficient code.


```
Connectivity"3-node triangle → 1-node point"
```

Mandatory information about

## DOF Handler

The dof-handler maps dofs to unique integer indices. A simple dof handler
that works for a nodal basis may be implemented in two steps. The first step,
an initialization procedure, has the be run only once. Here we assign each cell
type a disjoint integer range as large as the number of cells of this type in the mesh.
Now every

implemented by first assigning disjoint
integer ranges to each cell type and using the starting point of each interval as an offset

the number of cells per type and

$\sum_K cells_per_type(K)$

## Quadrature

The `Quadrature` module provides generic code for the approximation of integrals
on a mesh by means of a weighted sum of point values.

$\int_K f(\mathbb{x}) \mathrm{d}K \approx \sum_{j=1}^n w_j^n f(c_j^n)$

Evaluation of the approximation is by means of calling `integrate(f, Kis)`
with `f` the integrand $f$ in procedural form and `Kis` a set of
cells given as a `MeshFunction` of cell `Geometry` objects. Quadrature nodes and
weights are defined in the concrete mesh implementation which then may be used by the
generic algorithms in the `Quadrature` module.

## Matrix Assembler

## Vector Assembler

![Image of Yaktocat](../assets/tipi.svg)
