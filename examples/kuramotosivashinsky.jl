using BenchmarkTools
using FFTW
using GLMakie
using LinearAlgebra
using HDF5
using ProgressBars
using Random

using FourierPlayground, FourierPlayground.Grid, FourierPlayground.Domain

include("interpolation.jl")

"""
    Runs a simulaton of the Kuramoto-Sivashinksy equation.
"""
# reproducible noise
Random.seed!(123456789)
noise_const = 3e-2

# experiment setup
arraytype = Array
L = 22.0 # domain size / "reynolds" number
Ω = S¹(L)
N = 2^8  # number of gridpoints

# grid and wavenumbers
grid = FourierGrid(N, Ω, arraytype=arraytype)
nodes, wavenumbers = grid.nodes, grid.wavenumbers
x = nodes[1]
k = wavenumbers[1]

# construct wavenumber filter
# we filter out the smallest 1/3 of scales
kmax = maximum(k)
filter = @. abs(k) ≤ 0.5 * kmax # 2 / 3 * kmax

# differential operators
∂x = im * k
Δ = @. ∂x^2
Δ² = @. Δ^2
Δ⁻¹ = 1 ./ Δ
Δ⁻¹[1] = 0.0
Δx = x[2] - x[1]

# time step
c = 3
Δt = 0.1 * Δx / c
inv_op = @. 1 / (1 + Δt * Δ + Δt * Δ²)

# declare and initialize fields 
uⁿ⁺¹ = zeros(ComplexF64, N)
uⁿ = zeros(ComplexF64, N)
u²x = zeros(ComplexF64, N)
u² = zeros(ComplexF64, N)

# set up the FFTs
P! = plan_fft!(uⁿ, flags=FFTW.MEASURE)
P⁻¹! = plan_ifft!(uⁿ, flags=FFTW.MEASURE)

# nonlinear terms in KS equation
function nonlinear_term!(u²x, uⁿ, u², P!, P⁻¹!, ∂x, filter)
    # always assume u is in fourier space
    # use 2/3 rule for stability
    @. u² = filter * uⁿ
    P⁻¹! * u²
    @. u² = u² * u²
    P! * u²
    @. u²x = -0.5 * ∂x * u²

    return nothing
end

# initial condition
@. uⁿ = sin(2π / L * x) + 0.1 * cos(2 * 2π / L * x) + 0.1 * sin(11 * 2π / L * x)
P! * uⁿ

# set up time stepping and data storage for plotting and disk storage
substep = floor(Int, 0.2/Δt)
endtime = 20000
M = floor(Int, endtime / (substep * Δt))
plotmat = zeros(N, M)
save_N = 2 * 64
savemat = zeros(save_N, M)

# time stepping loop
tic = Base.time()
for i in ProgressBar(1:M)
    for j in 1:substep
        noiseterm = noise_const .* Δ⁻¹ .* (randn(size(Δ)) + im * randn(size(Δ)))
        nonlinear_term!(u²x, uⁿ, u², P!, P⁻¹!, ∂x, filter)
        @. uⁿ⁺¹ = inv_op * (uⁿ + Δt * u²x + √Δt * noiseterm)
        @. uⁿ = uⁿ⁺¹
    end
    P⁻¹! * uⁿ
    @. uⁿ = real(uⁿ)
    savemat[:, i] .= quick_interpolation_1d(uⁿ, M=save_N)
    plotmat[:, i] .= real.(uⁿ)
    P! * uⁿ
end
toc = Base.time()
println("The time for the simulation was ", toc - tic, " seconds")

# post processing
fig = Figure(resolution=(1000, 500))
ax = Axis(fig[1, 1])
ax_spec = Axis(fig[1, 2])
ax_inst = Axis(fig[1, 3])
contourf!(ax, plotmat[:, 1:substep:10000])
scatter!(ax_spec, log.(abs.(uⁿ) .+ eps(1.0)))
lines!(ax_inst, plotmat[:, end])
display(fig)

# store data to HDF5
if N == 32
    fid = h5open("ks_n32_low_res.hdf5", "w")
    println("saving low rez")
    fid["u"] = permutedims(savemat[:, :, :], (1, 3, 2))
    close(fid)
elseif N == 64
    fid = h5open("ks_n64_medium_res.hdf5", "w")
    println("saving medium rez")
    fid["u"] = permutedims(savemat[:, :, :], (1, 3, 2))
    close(fid)
elseif N == 256
    fid = h5open("ks_n256_high_res.hdf5", "w")
    println("saving high rez")
    fid["u"] = permutedims(savemat[:, :, :], (1, 3, 2))
    close(fid)
end
