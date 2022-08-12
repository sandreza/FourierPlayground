using FourierPlayground, Test, FFTW, LinearAlgebra
using FourierPlayground.Grid
import FourierPlayground.Grid: nodes, wavenumbers

@testset "nodes" begin
    x = nodes(10)
    @test length(nodes(10)) == 10
    @test x[1] == 0.0
    @test x[10] == 2.0 * pi - 1 / 10 * 2.0 * pi
end

@testset "wavenumbers" begin
    k = wavenumbers(11)
    @test length(k) == 11
    @test k[1] == 0.0
    @test k[2] == 1.0
    @test k[6] == -6.0
    @test k[end] == -1.0
end

@testset "fft" begin
    x = nodes(8)
    y = sin.(x)
    @test imag(fft(y)[2]) == -4.0
    @test imag(fft(y)[end]) == 4.0
    y = cos.(x)
    @test real(fft(y)[2]) == 4.0
    @test real(fft(y)[end]) == 4.0
end

@testset "Constructing the grid" begin
    grid = FourierGrid(4, Circle())
    k⃗ = grid.wavenumbers[]
    @test k⃗[1] == 0.0
    @test k⃗[2] == 1.0
    @test k⃗[3] == -2.0
    @test k⃗[4] == -1.0
    @test length(k⃗) == 4
    @test grid[1][1] == 0.0

    grid = FourierGrid(4, Circle(1π))
    k⃗ = grid.wavenumbers[]
    @test k⃗[1] == 0.0
    @test k⃗[2] == 2.0
    @test k⃗[3] == -4.0
    @test k⃗[4] == -2.0
    @test length(k⃗) == 4

    grid = FourierGrid((8, 8), Circle()^2)
    (; nodes, wavenumbers) = grid
    @test length(nodes) == 2
    @test length(wavenumbers) == 2
    @test all(wavenumbers[1][:] .== wavenumbers[2][:])
    @test all(LinearAlgebra.size(wavenumbers[1]) .== (8, 1))
    @test all(LinearAlgebra.size(wavenumbers[2]) .== (1, 8))

    grid = FourierGrid((3, 5, 7), Circle()^3)
    (; nodes, wavenumbers) = grid
    @test length(nodes) == 3
    @test length(wavenumbers) == 3
    @test all(LinearAlgebra.size(wavenumbers[1]) .== (3, 1, 1))
    @test all(LinearAlgebra.size(wavenumbers[2]) .== (1, 5, 1))
    @test all(LinearAlgebra.size(wavenumbers[3]) .== (1, 1, 7))

end