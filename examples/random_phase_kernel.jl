using KernelAbstractions
# using CUDAKernels

# only expect performance on the gpu

@kernel function random_phase_kernel!(field, @Const(A), @Const(𝓀ˣ), @Const(𝓀ʸ), @Const(x), @Const(y), @Const(φ), Nx, Ny)
    i, j = @index(Global, NTuple)

    tmp_sum = zero(eltype(field)) # create temporary array
    xx = x[i]
    yy = y[j]
    for ii in 1:Nx, jj in 1:Ny
        tmp_sum += A[ii, jj] * cos(𝓀ˣ[ii] * xx + 𝓀ʸ[jj] * yy + φ[ii, jj])
    end

    field[i, j] = tmp_sum
end


function stream_function!(field, A, 𝓀ˣ, 𝓀ʸ, x, y, φ)
    Nx = length(𝓀ˣ)
    Ny = length(𝓀ʸ)
    if typeof(field) <: Array
        kernel! = random_phase_kernel!(CPU(), 16)
        event = kernel!(field, A, 𝓀ˣ, 𝓀ʸ, x, y, φ, Nx, Ny, ndrange = size(field))
    else
        comp_stream = Event(CUDADevice())
        kernel! = random_phase_kernel!(CUDADevice(), 256)
        event = kernel!(field, A, 𝓀ˣ, 𝓀ʸ, x, y, φ, Nx, Ny, ndrange = size(field), dependencies = (comp_stream,))
    end
    return event
end
   
function φ_rhs_normal!(φ̇, φ, rng)
    randn!(rng, φ̇) 
    φ̇ .*= sqrt(1 / 12)
    return nothing
end
