
<a id='TipiFEM-Framework-for-the-Finite-Element-Method-1'></a>

# TipiFEM - Framework for the Finite Element Method


TipiFEM is a framework for the implementation of Finite Element Methods in julia.


<a id='Mesh-1'></a>

## Mesh


We define a mesh $\cal M$ as subdivision of a bounded set $\Omega \subset \mathbb{R}^d$ into open cells of dimension $< d$, where a cell is a topological space homeomorph to an open ball. For Finite Element Methods the cells are in almost all cases Polytopes or entities with the same combinatorial structure as Polytopes. The mesh is then used to represent the domain on which the problem we are trying to solve with our FEM solver is posed on.


The mesh dimension or simply dimension of a mesh is the largest dimension of its cells. The dimension of the ambient space $d$ of the mesh is called the world dimension.


  * Support for different element types

      * 3-noded triangular elements
      * 4-noded quadrilateral elements


```
test
```


```
world_dim
```


<a id='MeshInterface-1'></a>

### MeshInterface


immutable Edge <: Cell end dim(::Type{Edge}) = 1 face_count(::Type{Edge}, ::Type{Vertex}) = 2 facet(::Type{Edge}) = Vertex volume(geo::Geometry{Edge}) = point(geo, 2)[1] - point(geo, 1)[1]


TipiFEM contains a generic mesh implementation that may be used


__Simple1DMesh__


__PolytopalMesh__


<a id='MeshFunction-1'></a>

### MeshFunction


The central data-type used by the mesh is a `MeshFunction`. It is defined as a discrete function that maps cell-indices of cells with a fixed dimension to arbitrary objects. `MeshFunctions` are used to store the connectivity, material parameters or user defined attributes of mesh cells. Together a set of mesh functions may then define all properties of the cells in the mesh and give them an identity through their index.


**MeshTopology**


The topology of the mesh is represented by incidence relations stored as a $(d+1) \times (d+1)$ Matrix 𝕀 of `Connectivity`-valued `MeshFunction`s, where 𝕀$(d_1, d_2)$ stores the incidence relation between cells of dimension $d_1$ with those of dimension $d_2$.


<a id='MeshCell-1'></a>

### MeshCell


Since the cells of a mesh are stored implicitly through `MeshFunctions` their data-types also do not store any information. A concrete `MeshCell` (e.g. a `Polytope`) is therefore implemented as a singleton type in julia, whose only purpose is to allow distinguishing different types of cells. Intrinsic properties of mesh cells may then be defined by simple functions taking a concrete `MeshCell` as an argument. The dimension of a 3-node triangle for example is defined like this:


```
dim(::Polytope"3-node triangle") = 2
```


<a id='Connectivity-1'></a>

#### Connectivity


<a id='Geometry-1'></a>

#### Geometry


<a id='Misc-1'></a>

#### Misc


**Dim, Codim**


The `Dim` and `Codim` types are used to represent dimension and codimension information. Since in most situations the dimension and the codimension is already known at compile time these types are implemented as singleton types that store the value of the (co)dimension as a parametric type (note that the value is an integer and not a type). This allows us to dispatch on the (co)dimension, which is especially handy in cases, where the knowledge of the (co)dimension at compile time allows generation of efficient code.


```
Connectivity"3-node triangle → 1-node point"
```


Mandatory information about


<a id='DOF-Handler-1'></a>

## DOF Handler


<a id='Matrix-Assembler-1'></a>

## Matrix Assembler


<a id='Vector-Assembler-1'></a>

## Vector Assembler


![Image of Yaktocat](../assets/tipi.svg)

