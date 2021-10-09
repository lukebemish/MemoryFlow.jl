module MemoryFlow
export SimpleNode, AbstractNode
export solvediffeq, makediffeq, treeparse

using LSODA, DiffEqBase

abstract type AbstractNode end

struct SimpleNode <: AbstractNode
    # we need checks for this function!
    f::Function # f(this_node,input_nodes) -> ((t,x,xs) -> dx/dt)
    inputs::Vector
end

function treeparse(nodes::Vector, done::Vector)
    newdone=done
    for i=nodes
        if i ∉ done
            outpair = i=>length(newdone)+1
            newdone = [newdone;outpair]
        end
        for j=i.inputs
            if j ∉ done
                newdone = treeparse([j],newdone)
            end
        end
    end
    newdone
end

function makediffeq(parsedtree::Vector)
    fs = []
    for node=getproperty.(parsedtree,:first)
        ids = [parsedtree[i] for i=node.inputs]
        idthis = parsedtree[node]
        nf = node.f(node,node.inputs)
        f = (t,x) -> nf(t,x[idthis],x[ids])
        push!(fs,f)
    end
    iterer = enumerate(fs)
    return (dx,x,p,t) -> [dx[i]=f(t,x) for (i,f) in iterer], length(fs)
end

function solvediffeq(neteq, size; x_init = 0.5, tmax=1.)
    neteq! = diffeqtuple[1]
    size = diffeqtuple[2]
    tspan = (0.,tmax)
    x0 = fill(x_init,size)

    prob = ODEProblem(neteq!,x0,tspan)
    solve(prob,lsoda())
end

include("nodes.jl")

end