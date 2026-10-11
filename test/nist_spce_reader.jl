"""
Reader for the NIST SPC/E sample configurations (test/data/nist_spce) plus the SPC/E parameters
and CODATA-2010 Coulomb constant used by the NIST reference calculations. Shared by
test/test_electrostatics_nist.jl and dev/wolf_parameter_scan.jl.
"""

const NIST_DIR = joinpath(@__DIR__, "data", "nist_spce")

# SPC/E model and constants exactly as on the NIST pages (CODATA 2010)
const SPCE_QH = 0.42380                     # e
const SPCE_SIGMA = 3.16555789                # Å
const SPCE_EPS = 78.19743111                 # K
const E_CHARGE = 1.602176565e-19             # C
const EPS0 = 8.854187817e-12                 # C²/(J m)
const KB = 1.3806488e-23                     # J/K
const KE_K_ANGSTROM = E_CHARGE^2 / (4π * EPS0 * KB) * 1e10   # K·Å / e²

function read_nist_config(path::String)
    lines = readlines(path)
    L = parse(Float64, split(lines[1])[1])
    M = parse(Int, strip(lines[2]))
    n = 3 * M
    pos = zeros(Float64, 3, n)
    q = zeros(Float64, n)
    σ = zeros(Float64, n)
    ε = zeros(Float64, n)
    mol = zeros(Int, n)
    for k in 1:n
        f = split(lines[2 + k])
        idx = parse(Int, f[1])
        @assert idx == k
        pos[1, k] = parse(Float64, f[2])
        pos[2, k] = parse(Float64, f[3])
        pos[3, k] = parse(Float64, f[4])
        el = f[5]
        if el == "O"
            q[k] = -2 * SPCE_QH; σ[k] = SPCE_SIGMA; ε[k] = SPCE_EPS
        elseif el == "H"
            q[k] = SPCE_QH
        else
            error("unknown element $el")
        end
        mol[k] = (k - 1) ÷ 3 + 1   # file order is O, H, H per molecule
    end
    return L, M, pos, q, σ, ε, mol
end

