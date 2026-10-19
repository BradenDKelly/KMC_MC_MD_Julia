# Loader of the GROMACS reference configurations test/data/gromacs_<solute>_spce500.gro (6 decimals; solute first, then
# SPC/E) into a RigidSystem of MolSim's model of the same solute (used by test_solutes_gromacs.jl and test_pme_ekmc.jl).
const MCq = MolSim.MC

function load_solute_water_gro(sym, path; coulomb=:rf, α_A=0.20)
    m = MCq.spce_solute_model(sym; coulomb=coulomb, epsilon_rf=Inf, α_per_angstrom=α_A, cutoff=:site, rc_angstrom=10.0)
    σA = m.σ_angstrom; lines = readlines(path); nat = parse(Int, strip(lines[2]))
    box = parse.(Float64, split(lines[end])[1:3]); L = box[1] * 10 / σA
    X = zeros(3, nat)
    for k in 1:nat
        l = lines[2 + k]; X[:, k] .= parse.(Float64, (l[21:31], l[32:42], l[43:53])) .* (10 / σA)
    end
    tw, ts = m.templates[1], m.templates[2]; nw = (nat - ts.n_sites) ÷ 3
    sys = MCq.RigidSystem(m.templates, vcat(2, fill(1, nw)), L, 1.0, Xoshiro(1))
    Y = X[:, 1:ts.n_sites]
    for a in 2:ts.n_sites, d in 1:3; Y[d, a] = Y[d, 1] + MCq._min_image(Y[d, a] - Y[d, 1], L); end
    MCq.set_conformation!(sys, 1, Y); MCq.set_molecule!(sys, 1, (Y * ts.site_mass) ./ sum(ts.site_mass), sys.quat[:, 1])
    for w in 1:nw
        W = X[:, ts.n_sites + 3(w - 1) .+ (1:3)]
        for a in 2:3, d in 1:3; W[d, a] = W[d, 1] + MCq._min_image(W[d, a] - W[d, 1], L); end
        MCq.molecule_from_sites!(sys, 1 + w, W)
    end
    return m, sys, nw
end
