module MemoryFlow
export make_nodes, eval_nodes, update_node

using LSODA, DiffEqBase

abstract type AbstractNode end

"""
    struct SimpleNode <: AbstractNode
        inputs::Vector
        val::Real
        starting_val::Real
    end
The bare minimum for a functional node. All nodes should be an AbstractNode.

`inputs` is the Vector of input nodes.

`val` is the current stored value of the node.

`starting_val` is the value the node should start at during differential
equation solving.
"""
mutable struct SimpleNode <: AbstractNode
    inputs::Vector
    val::Float64
    starting_val::Float64
end

function equation_constructor(this_node::AbstractNode,input_nodes)
    (t,x,xs) -> 0
end

function treeparse(nodes::Vector, done::Vector)
    newdone=done
    for i=nodes
        if i ∉ getproperty.(newdone,:first)
            outpair = i=>length(newdone)+1
            push!(newdone,outpair)
        end
        for j=i.inputs
            if j ∉ getproperty.(newdone,:first)
                newdone = treeparse([j],newdone)
            end
        end
    end
    newdone
end

function makediffeq(vectortree::Vector)
    parsedtree = IdDict(vectortree)
    fs = []
    nodes = keys(parsedtree)
    for node=nodes
        ids = [parsedtree[i] for i=node.inputs]
        idthis = parsedtree[node]
        nf = equation_constructor(node,node.inputs)
        f = (t,x) -> nf(t,x[idthis],x[ids])
        push!(fs,f)
    end
    iterer = enumerate(fs)
    return (dx,x,p,t) -> [dx[i]=f(t,x) for (i,f) in iterer], nodes
end

function solvediffeq(neteq!, nodes;tmax=1.)
    tspan = (0.,tmax)
    x0 = [i.starting_val for i=nodes]

    prob = ODEProblem(neteq!,x0,tspan)
    solve(prob,lsoda())
end

function make_nodes end

function update_val!(node::AbstractNode, val::Float64)
    node.val = val
end

function eval_nodes(nodes::Vector;kwargs...)
    tree = treeparse(nodes,[])
    eq, ns = makediffeq(tree)
    sol = solvediffeq(eq,ns;kwargs...)
    last_sol = last(sol.u)
    for (i,n)=enumerate(ns)
        update_val!(n,last_sol[i])
    end
    return [i.val for i=nodes]
end

function update_node(node::AbstractNode)
    # This does nothing; most nodes don't need updating.
end

include("nodes.jl")

end