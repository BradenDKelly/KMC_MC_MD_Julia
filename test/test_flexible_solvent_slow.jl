# DESIGN §14 V3: liquid OPLS-AA ethanol (flexible, reaction field), eKMC with flexible solvent L-moves (translation, rotation,
# single-site, torsion slots; no accept/reject) vs Metropolis MC with translation, rotation, site and torsion moves:
# intermolecular U/N, ⟨U_intra⟩ and P within 3 combined standard errors (fixed seeds).
using Test, Random
using MolSim
const MCs = MolSim.MC
fss_within3σ(a, ea, b, eb) = abs(a - b) <= 3 * sqrt(ea^2 + eb^2)

"Compress a random low-density start to `L` by COM rescaling in steps (minimising after each) — a set-up protocol only."
function compress_to!(sys, ff, L; steps=12, nsteps=150)
    f = (L / sys.L)^(1 / steps)
    for _ in 1:steps
        sys.L *= f
        for i in 1:sys.n_mol
            MCs.set_molecule!(sys, i, sys.com[:, i] .* f, sys.quat[:, i])
        end
        MCs.minimize_rigid!(sys, ff; nsteps=nsteps)
    end
    sys.L = L
    return sys
end

@testset "Liquid OPLS-AA ethanol (RF): eKMC + flexible solvent L-moves vs Metropolis MC (U, ⟨U_intra⟩, P)" begin
    N = 64; T = 298.0
    m = MCs.spce_solute_model(:ethanol_flex; coulomb=:rf, epsilon_rf=Inf, cutoff=:site, rc_angstrom=9.0)
    t = m.templates[2]; ff = m.ff; β = m.ϵ_kelvin / T
    ρ = 0.785 / 46.07 * 6.02214076e23 / 1e24
    L = cbrt(N / ρ) / m.σ_angstrom
    sys = MCs.RigidSystem([t], fill(1, N), 2L, β, Xoshiro(21)); MCs.random_rigid_system!(sys, ff)
    compress_to!(sys, ff, L)
    res = MCs.build_reservoir([t], 1, ff, β, Xoshiro(5); M=4000, burnin_sweeps=500, stride_sweeps=20, δ=0.02)
    nt = length(MCs.rotatable_bonds(t))
    δ, δθ, δs, φ = 0.05, 0.15, 0.01, 0.5                         # step sizes: efficiency choices
    MCs.run_rigid_mc!(sys, ff; sweeps=1000, δ=δ, δθ=δθ, sample_every=100, block_size=5, reservoirs=[res],
                      δ_site=δs, torsion_φ=φ)                    # common equilibration
    mc = MCs.run_rigid_mc!(deepcopy(sys), ff; sweeps=12_000, δ=δ, δθ=δθ, sample_every=2, block_size=200,
                           reservoirs=[res], δ_site=δs, torsion_φ=φ)
    ν = 1e-3
    events = 3_000_000
    ek = @test_logs min_level=Base.CoreLogging.Warn MCs.run_rigid_ekmc!(deepcopy(sys), ff; events=events, sample_every=20,
            block_events=events ÷ 25, burnin_events=events ÷ 10, debug_check_every=events ÷ 6, reservoirs=[res],
            local_solvent=true, solvent_flexible=true, solvent_nu=ν, mc_δ=δ, mc_δθ=δθ, solvent_site_a=δs, solvent_φ=φ)
    S = ek.slmoves
    kJ = m.ϵ_kelvin * 0.0083144626
    @info "liquid ethanol (RF, N=64): MC vs eKMC + flexible solvent L-moves (kJ/mol)" MCs.mean(mc.U) * kJ MCs.stderr(mc.U) * kJ MCs.mean(ek.U) * kJ MCs.stderr(ek.U) * kJ MCs.mean(mc.U_intra) * kJ MCs.stderr(mc.U_intra) * kJ MCs.mean(ek.U_intra) * kJ MCs.stderr(ek.U_intra) * kJ MCs.mean(mc.P) MCs.stderr(mc.P) MCs.mean(ek.P) MCs.stderr(ek.P) S.kinds S.n_move S.n_refresh ek.relocations nt
    @test nt >= 1 && MCs.LM_D in S.kinds && all(S.n_move[[MCs.LM_T, MCs.LM_R, MCs.LM_S, MCs.LM_D]] .> 0)
    @test fss_within3σ(MCs.mean(mc.U), MCs.stderr(mc.U), MCs.mean(ek.U), MCs.stderr(ek.U))
    @test fss_within3σ(MCs.mean(mc.U_intra), MCs.stderr(mc.U_intra), MCs.mean(ek.U_intra), MCs.stderr(ek.U_intra))
    @test fss_within3σ(MCs.mean(mc.P), MCs.stderr(mc.P), MCs.mean(ek.P), MCs.stderr(ek.P))
end
