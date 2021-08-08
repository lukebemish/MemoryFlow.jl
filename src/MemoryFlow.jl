struct SimpleMemoryLayer :> AbstractMemoryLayer
    # Add some sort of check that f is valid, because if it's not, then
    # there will be some serious issues.
    f::Function # f(self_values,layers,t) -> Vector{dy/dt} (size is outsize)
    inputs::Vector
    outsize::Real
end

function treeparse(layer::AbstractMemoryLayer, done::Vector)
    outpair = layer => layer.outsize
    newdone = [done;layer]
    tocheck = filter((x) -> x ∉ done, layer.inputs)
    return [outpair;[treeparse(l,newdone) for l=tocheck]...]
end

function makediffeq(parsedtree::Vector)
    keys=getproperty.(parsedtree,:first)
    values=getproperty.(parsedtree,:second)
    useful = Dict(keys[i]=>(values[i],i) for i=1:size(keys))
    outeq=[]
    for layer=keys
        idxs = [useful[i][2]:(useful[i][2]+useful[i][1]-1) for i in layer.inputs]
        sidxs = useful[layer][2]:(useful[layer][2]+useful[layer][1]-1)
        f = (x,t) -> layer.f(x[sidxs],x[(idxs...)...],t)
        push!(outeq,f)
    end
    return (x,t) -> [([f(x,t) for f in outeq]...)...]
end