using LinearAlgebra

struct RozellLCANode <: AbstractNode
    f::Function
    inputs::Vector
    competitive_key::Vector # vector of places in input where input is anti-hebbian
    biases::Vector # length is length(input) - length(competitive_key)
end

function rozell_lca_f_assembler(this_node,input_nodes)
    ϕ = this_node.biases
    τ = 1
    Gs = [i.biases ⋅ ϕ for i=input_nodes[this_node.competitive_key]]
    in_ids = filter(x -> x ∉ this_node.competitive_key, 1:length(input_nodes))
    comp_iter = enumerate(this_node.competitive_key)
    (t,x,xs) -> (ϕ ⋅ xs[in_ids] - x - sum([Gs[i]*xs[j] for (i,j)=comp_iter]))/τ
end

function make_rozell_lca_layer(size::Integer,inputs::Vector)
    nodes = []
    for i=1:size
        init_biases = rand(Float64,length(inputs))
        normalize!(init_biases)
        node = RozellLCANode(rozell_lca_f_assembler,inputs,[],init_biases)
        push!(nodes,node)
    end
    for i=1:size
        to_add = filter(x->x!=nodes[i],nodes)
        push!(nodes[i].inputs,to_add...)
        push!(nodes[i].competitive_key,length(inputs)+1:length(inputs)+size-1)
    end
    nodes
end

export make_rozell_lca_layer

# This is an implementation based on the network described here:
#
# Rozell, Christopher J., et al. "Sparse coding via thresholding and local
# competition in neural circuits." Neural computation 20.10 (2008): 2526-2563.
