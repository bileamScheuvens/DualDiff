import Base: +, -, *, /, promote_rule, convert, show, abs
using LinearAlgebra

struct Dual <: Number
    re_eps::Tuple{Float64, Float64}
end
# get real part
re(x::Number) = x
re(x::Dual) = x.re_eps[1]
# get epsilon part
ϵ(x::Number) = 0
ϵ(x::Dual) = x.re_eps[2]
# manage type conversion
convert(::Type{Dual}, x::Real) = Dual((x, zero(x)))
promote_rule(::Type{<:Dual}, ::Type{<:Number}) = Dual
# manage display
show(io::IO, x::Dual) = print(io, "$(re(x)) + $(ϵ(x)) ϵ")

# Basic Operations
+(a::Dual, b::Dual) = (a.re_eps .+ b.re_eps) |> Dual
-(a::Dual, b::Dual) = (a.re_eps .- b.re_eps) |> Dual
*(a::Dual, b::Dual) = (re(a) * re(b), re(a) * ϵ(b) + ϵ(a) * re(b)) |> Dual
/(a::Dual, b::Dual) = (re(a) / re(b), (ϵ(a) * re(b) - re(a) * ϵ(b)) / ϵ(b)^2) |> Dual
Base.sin(x::Dual) = Dual((sin(re(x)), cos(re(x)) * ϵ(x)))
abs(x::Dual) = abs.(x.re_eps) |> Dual


# Get derivative
push_forward(f::Function, x::Number) = f(Dual((x, 1))) |> ϵ

function push_forward(f::Function, x::Vector{<:Number})
    dimin = length(x)
    duals = map(Dual, zip(repeat(x, 1, dimin), LinearAlgebra.I(dimin)))
    eachrow(duals) |> 
        x -> map(f, x...) .|> 
        collect |>
        x -> permutedims(hcat(x...)) .|>
        ϵ |>
        transpose |>
        collect
end

function push_forward(f::Function, x::Matrix)
    map(x->push_forward(f, collect(x)), eachrow(x))
end

f(a,b) = a^2, b, 2

push_forward(f, [3,4])

push_forward(f, [1 2; 3 4])

j(x,y) = x^2*y, 5x+sin(y)

push_forward(j, [1,1])
push_forward(j, [1 1; 2 2])[2]