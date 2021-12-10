using LinearAlgebra

mutable struct InputNode <: AbstractNode
    inputs::Vector
    val::Float64
    starting_val::Float64
end

export set_input_val
function set_input_val!(node::InputNode, val::Real)
    node.starting_val=val
    node.val=val
end

function make_nodes(type::Val{:Input},size::Integer)
    [InputNode([],0,0) for i=1:size]
end

export set_input_val!

function Base.show(io::IO, n::InputNode)
    print(io, "MemoryFlow.InputNode: ", n.starting_val)
end

mutable struct RozellLCANode <: AbstractNode
    inputs::Vector{AbstractNode}
    val::Float64
    starting_val::Float64
    competitive_key::Vector # vector of places in input where input is anti-hebbian
    biases::Vector # length is length(input) - length(competitive_key)
end

function Base.show(io::IO, n::RozellLCANode)
    print(io, "MemoryFlow.RozellLCANode: ", length(n.inputs)-length(n.competitive_key),
        ", ",length(n.competitive_key), ", ", n.val)
end

thresval = 0.2

function rozell_thres(val)
    (val-thresval)*(val>thresval)
end

function equation_constructor(this_node::RozellLCANode,input_nodes)
    ϕ = this_node.biases
    τ = 1
    Gs = [i.biases ⋅ ϕ for i=input_nodes[this_node.competitive_key]]
    in_ids = filter(x -> x ∉ this_node.competitive_key, 1:length(input_nodes))
    comp_iter = enumerate(this_node.competitive_key)
    (t,x,xs) -> ((ϕ ⋅ xs[in_ids]) - x - sum([Gs[i]*rozell_thres(xs[j]) for (i,j)=comp_iter]))/τ
end

function make_rozell_lca_layer(size::Integer,inputs::Vector)
    nodes = []
    for i=1:size
        init_biases = rand(Float64,length(inputs))
        normalize!(init_biases)
        node = RozellLCANode(
            inputs,0,0,[],init_biases)
        push!(nodes,node)
    end
    for i=1:size
        to_add = filter(x->x!=nodes[i],nodes)
        push!(nodes[i].inputs,to_add...)
        push!(nodes[i].competitive_key,(length(inputs)+1:length(inputs)+size-1)...)
    end
    nodes
end

function update_node!(node::RozellLCANode; β=0.02, kwargs...)
    # Node updating or Hebbian learning goes here
    # To start with, I'll jsut have some simple Hebbian learning
    in_ids = filter(x -> x ∉ node.competitive_key, 1:length(node.inputs))
    dq = β * node.val * (node[in_ids] - node.biases)
    node.biases += dq
end

function make_nodes(type::Val{:RozellLCA},size::Integer,inputs::Vector,kwargs...)
    return make_rozell_lca_layer(size,inputs;kwargs...)
end

function update_val!(node::RozellLCANode, val::Float64)
    node.val = rozell_thres(val)
end

# This is an implementation based on the network described here:
#
# Rozell, Christopher J., et al. "Sparse coding via thresholding and local
# competition in neural circuits." Neural computation 20.10 (2008): 2526-2563.
