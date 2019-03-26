import Base: push!, resize!, convert, length

mutable struct Triplets{T <: Real}
  I::Vector{Int}
  J::Vector{Int}
  V::Vector{T}
  length::Int

  Triplets{T}(N::Int) where T <: Real = new{T}(Vector{Int}(undef, N), Vector{Int}(undef, N), Vector{T}(undef, N), 0)
end

function push!(triplets::Triplets{T}, i::Int, j::Int, v::T) where T <: Real
  triplets.length+=1
  triplets.I[triplets.length] = i
  triplets.J[triplets.length] = j
  triplets.V[triplets.length] = v
  triplets
end

function resize!(triplets::Triplets, n::Int)
  resize!(I, n)
  resize!(J, n)
  resize!(V, n)
end

length(triplets::Triplets) = triplets.length

import SparseArrays: sparse
function convert(SparseMatrixCSC, triplets::Triplets)
  l=length(triplets)
  sparse(triplets.I[1:l], triplets.J[1:l], triplets.V[1:l])
end

import SparseArrays: sparse
function sparse(triplets::Triplets, m::Int, n::Int)
  l=length(triplets)
  sparse(view(triplets.I, 1:l), view(triplets.J, 1:l), view(triplets.V, 1:l), m, n)
end
