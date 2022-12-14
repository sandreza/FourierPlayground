using KernelAbstractions
# using CUDAKernels

# only expect performance on the gpu

@kernel function random_phase_kernel!(field, @Const(A), @Const(πΛ£), @Const(πΚΈ), @Const(x), @Const(y), @Const(Ο), Nx, Ny)
    i, j = @index(Global, NTuple)

    tmp_sum = zero(eltype(field)) # create temporary array
    xx = x[i]
    yy = y[j]
    for ii in 1:Nx, jj in 1:Ny
        tmp_sum += A[ii, jj] * cos(πΛ£[ii] * xx + πΚΈ[jj] * yy + Ο[ii, jj])
    end

    field[i, j] = tmp_sum
end


function stream_function!(field, A, πΛ£, πΚΈ, x, y, Ο)
    Nx = length(πΛ£)
    Ny = length(πΚΈ)
    if typeof(field) <: Array
        kernel! = random_phase_kernel!(CPU(), 16)
        event = kernel!(field, A, πΛ£, πΚΈ, x, y, Ο, Nx, Ny, ndrange = size(field))
    else
        comp_stream = Event(CUDADevice())
        kernel! = random_phase_kernel!(CUDADevice(), 256)
        event = kernel!(field, A, πΛ£, πΚΈ, x, y, Ο, Nx, Ny, ndrange = size(field), dependencies = (comp_stream,))
    end
    return event
end
   
function Ο_rhs_normal!(ΟΜ, Ο, rng)
    randn!(rng, ΟΜ) 
    ΟΜ .*= sqrt(1 / 12)
    return nothing
end
