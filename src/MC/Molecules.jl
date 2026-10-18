"""
Rigid molecule support for Lennard-Jones Monte Carlo.
Additive extension - does not modify atomic LJ behavior.
"""

using Random
using StaticArrays
using LinearAlgebra

"""
    MoleculeTemplate

Template defining a rigid molecule structure.
Sites are defined in body frame (relative to COM).
"""
struct MoleculeTemplate
    n_sites::Int
    sites_body::Matrix{Float64}  # 3 x n_sites, positions in body frame
    site_σ::Vector{Float64}      # LJ σ for each site (can be all 1.0)
    site_ϵ::Vector{Float64}      # LJ ϵ for each site (can be all 1.0)
end

"""
    MoleculeState

State of a single rigid molecule.
"""
mutable struct MoleculeState
    com::MVector{3,Float64}      # Center of mass position
    quat::MVector{4,Float64}     # Orientation quaternion (w, x, y, z)
    template_idx::Int             # Index into molecule templates array
end

"""
    MolecularSystem

System containing both atoms and molecules.
Atoms are stored as before in pos matrix.
Molecules are stored separately.
"""
mutable struct MolecularSystem
    # Atomic particles (existing)
    n_atoms::Int
    atom_pos::Matrix{Float64}     # 3 x n_atoms
    
    # Molecules
    n_molecules::Int
    molecules::Vector{MoleculeState}
    templates::Vector{MoleculeTemplate}
    
    # Global site mapping: for each site, which molecule (or -1 if atom)
    # site_to_molecule[i] = molecule index, or -1 if atom
    # site_to_site_idx[i] = site index within molecule, or atom index
    n_sites_total::Int
    site_to_molecule::Vector{Int}
    site_to_site_idx::Vector{Int}
    
    # Global site positions (computed from molecules + atoms)
    site_pos::Matrix{Float64}     # 3 x n_sites_total
    
    # Cell list (for all sites)
    cl::CellList
    
    # RNG and scratch
    rng::Xoshiro
    scratch_dr::MVector{3,Float64}
    
    # Acceptance tracking
    atom_accepted::Int
    atom_attempted::Int
    mol_trans_accepted::Int
    mol_trans_attempted::Int
    mol_rot_accepted::Int
    mol_rot_attempted::Int
    cbmc_insert_accepted::Int
    cbmc_insert_attempted::Int
    cbmc_delete_accepted::Int
    cbmc_delete_attempted::Int
end

"""
    quaternion_to_rotation_matrix(q)

Convert quaternion (w, x, y, z) to 3x3 rotation matrix.
Explicitly normalizes the quaternion for numerical robustness.
"""
function quaternion_to_rotation_matrix(q::AbstractVector{Float64})
    w, x, y, z = q[1], q[2], q[3], q[4]
    
    # Normalize quaternion for numerical robustness
    n2 = w*w + x*x + y*y + z*z
    invn = inv(sqrt(n2))
    w *= invn
    x *= invn
    y *= invn
    z *= invn
    
    # static 3×3 (no heap allocation: this runs for every site-offset update, i.e. every candidate of every event)
    return @SMatrix [1 - 2*(y*y + z*z)   2*(x*y - w*z)       2*(x*z + w*y);
                     2*(x*y + w*z)       1 - 2*(x*x + z*z)   2*(y*z - w*x);
                     2*(x*z - w*y)       2*(y*z + w*x)       1 - 2*(x*x + y*y)]
end

"""
    update_site_positions!(sys)

Update global site positions from molecule states and atom positions.
"""
function update_site_positions!(sys::MolecularSystem)
    # Copy atom positions
    @inbounds for i in 1:sys.n_atoms
        sys.site_pos[1, i] = sys.atom_pos[1, i]
        sys.site_pos[2, i] = sys.atom_pos[2, i]
        sys.site_pos[3, i] = sys.atom_pos[3, i]
    end
    
    # Compute molecule site positions
    site_idx = sys.n_atoms
    @inbounds for mol_idx in 1:sys.n_molecules
        mol = sys.molecules[mol_idx]
        template = sys.templates[mol.template_idx]
        
        # Get rotation matrix
        R = quaternion_to_rotation_matrix(mol.quat)
        
        # Transform each site
        for site_local_idx in 1:template.n_sites
            site_idx += 1
            
            # Body frame position
            r_body = template.sites_body[:, site_local_idx]
            
            # Rotate
            r_rot = R * r_body
            
            # Add COM
            sys.site_pos[1, site_idx] = mol.com[1] + r_rot[1]
            sys.site_pos[2, site_idx] = mol.com[2] + r_rot[2]
            sys.site_pos[3, site_idx] = mol.com[3] + r_rot[3]
        end
    end
end

"""
    uniform_random_quaternion(rng)

Generate Haar-uniform random quaternion on SO(3) using Shoemake algorithm.
This is the industry-standard method for uniform orientation sampling in molecular simulation.
"""
function uniform_random_quaternion(rng::AbstractRNG)::MVector{4,Float64}
    u1 = rand(rng)
    u2 = rand(rng)
    u3 = rand(rng)
    
    s1 = sqrt(1.0 - u1)
    s2 = sqrt(u1)
    
    θ1 = 2π * u2
    θ2 = 2π * u3
    
    x = s1 * sin(θ1)
    y = s1 * cos(θ1)
    z = s2 * sin(θ2)
    w = s2 * cos(θ2)
    
    return MVector{4,Float64}(w, x, y, z)
end

"""
    create_single_site_molecule_template()

Create a single-site molecule template (equivalent to atom).
"""
function create_single_site_molecule_template()::MoleculeTemplate
    sites_body = zeros(Float64, 3, 1)
    site_σ = [1.0]
    site_ϵ = [1.0]
    return MoleculeTemplate(1, sites_body, site_σ, site_ϵ)
end

"""
    create_diatomic_molecule_template(bond_length::Float64=1.0)

Create a diatomic molecule template with bond length.
"""
function create_diatomic_molecule_template(bond_length::Float64=1.0)::MoleculeTemplate
    sites_body = zeros(Float64, 3, 2)
    sites_body[1, 1] = -bond_length / 2.0
    sites_body[1, 2] = bond_length / 2.0
    site_σ = [1.0, 1.0]
    site_ϵ = [1.0, 1.0]
    return MoleculeTemplate(2, sites_body, site_σ, site_ϵ)
end

"""
    init_molecular_system(atom_pos, molecules, templates, L, rc, seed)

Initialize molecular system from atoms and molecules.
"""
function init_molecular_system(
    atom_pos::Matrix{Float64},
    molecules::Vector{MoleculeState},
    templates::Vector{MoleculeTemplate},
    L::Float64,
    rc::Float64,
    seed::Int
)::MolecularSystem
    n_atoms = size(atom_pos, 2)
    n_molecules = length(molecules)
    
    # Count total sites
    n_sites_total = n_atoms
    for mol in molecules
        template = templates[mol.template_idx]
        n_sites_total += template.n_sites
    end
    
    # Build site mapping
    site_to_molecule = Vector{Int}(undef, n_sites_total)
    site_to_site_idx = Vector{Int}(undef, n_sites_total)
    
    # Atoms map to -1
    for i in 1:n_atoms
        site_to_molecule[i] = -1
        site_to_site_idx[i] = i
    end
    
    # Molecules map to molecule index
    site_idx = n_atoms
    for mol_idx in 1:n_molecules
        mol = molecules[mol_idx]
        template = templates[mol.template_idx]
        for site_local_idx in 1:template.n_sites
            site_idx += 1
            site_to_molecule[site_idx] = mol_idx
            site_to_site_idx[site_idx] = site_local_idx
        end
    end
    
    # Allocate site positions
    site_pos = zeros(Float64, 3, n_sites_total)
    
    # Create cell list
    cl = CellList(n_sites_total, L, rc)
    
    # Create system
    sys = MolecularSystem(
        n_atoms, atom_pos,
        n_molecules, molecules, templates,
        n_sites_total, site_to_molecule, site_to_site_idx,
        site_pos, cl,
        Xoshiro(seed), MVector{3,Float64}(0.0, 0.0, 0.0),
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    )
    
    # Update site positions
    update_site_positions!(sys)
    
    # Rebuild cell list
    rebuild_cells_molecular!(sys, L)
    
    return sys
end

"""
    rebuild_cells_molecular!(sys, L)

Rebuild cell list for molecular system.
"""
function rebuild_cells_molecular!(sys::MolecularSystem, L::Float64)
    # Update site positions first
    update_site_positions!(sys)
    
    # Clear all head pointers
    fill!(sys.cl.head, 0)
    fill!(sys.cl.next, 0)
    
    N = sys.n_sites_total
    ncell = sys.cl.ncell
    pos = sys.site_pos
    
    # Assign sites to cells
    @inbounds for i in 1:N
        x, y, z = pos[1, i], pos[2, i], pos[3, i]
        # Wrap coordinates to [0, L)
        x_wrapped = x - L * floor(x / L)
        y_wrapped = y - L * floor(y / L)
        z_wrapped = z - L * floor(z / L)
        
        cell_idx = get_cell(x_wrapped, y_wrapped, z_wrapped, L, ncell)
        sys.cl.cell_of[i] = cell_idx
        sys.cl.next[i] = sys.cl.head[cell_idx]
        sys.cl.head[cell_idx] = i
    end
    
    return nothing
end

# Note: Since Molecules.jl is included in MC module, types from other included files
# are in the same namespace. We reference them directly without imports.

"""
    molecular_local_energy(site_i::Int, sys::MolecularSystem, p::LJParams)::Float64

Compute local energy for site i (sum over neighbors within rc).
Excludes intramolecular interactions (sites from same molecule).
"""
function molecular_local_energy(site_i::Int, sys::MolecularSystem, p::LJParams)::Float64
    energy = 0.0
    N = sys.n_sites_total
    L = sys.cl.L
    rc2 = p.rc2
    pos = sys.site_pos
    
    # Get site i position and molecule
    pix = pos[1, site_i]
    piy = pos[2, site_i]
    piz = pos[3, site_i]
    mol_i = sys.site_to_molecule[site_i]
    
    # Loop over all other sites (same structure as molecular_total_energy for consistency)
    @inbounds for j in 1:N
        if j != site_i
            mol_j = sys.site_to_molecule[j]
            
            # Skip intramolecular interactions
            if mol_i == mol_j
                continue
            end
            
            # Compute distance vector (same convention as molecular_total_energy)
            dr_x = pos[1, j] - pix
            dr_y = pos[2, j] - piy
            dr_z = pos[3, j] - piz
            
            # Apply minimum image convention (same as molecular_total_energy)
            L_half = L / 2.0
            if dr_x > L_half
                dr_x = dr_x - L
            elseif dr_x < -L_half
                dr_x = dr_x + L
            end
            if dr_y > L_half
                dr_y = dr_y - L
            elseif dr_y < -L_half
                dr_y = dr_y + L
            end
            if dr_z > L_half
                dr_z = dr_z - L
            elseif dr_z < -L_half
                dr_z = dr_z + L
            end
            
            r2 = dr_x*dr_x + dr_y*dr_y + dr_z*dr_z
            
            if r2 < rc2 && r2 > 0.0
                # Use mixed pair routine for consistency with atomic paths
                # For now, use type 1 (single-component default)
                # TODO: Support per-site types from molecule templates
                type_i = 1
                type_j = 1
                energy += lj_pair_u_from_r2_mixed(r2, type_i, type_j, p)
            end
        end
    end
    
    return energy
end

"""
    molecular_total_energy(sys::MolecularSystem, p::LJParams)::Float64

Compute total intermolecular energy (excludes intramolecular terms).
"""
function molecular_total_energy(sys::MolecularSystem, p::LJParams)::Float64
    energy = 0.0
    N = sys.n_sites_total
    L = sys.cl.L
    rc2 = p.rc2
    pos = sys.site_pos
    
    @inbounds for i in 1:N
        mol_i = sys.site_to_molecule[i]
        for j in (i+1):N
            mol_j = sys.site_to_molecule[j]
            
            # Skip intramolecular interactions
            if mol_i == mol_j
                continue
            end
            
            # Compute distance vector
            dr_x = pos[1, j] - pos[1, i]
            dr_y = pos[2, j] - pos[2, i]
            dr_z = pos[3, j] - pos[3, i]
            
            # Apply minimum image convention
            L_half = L / 2.0
            if dr_x > L_half
                dr_x = dr_x - L
            elseif dr_x < -L_half
                dr_x = dr_x + L
            end
            if dr_y > L_half
                dr_y = dr_y - L
            elseif dr_y < -L_half
                dr_y = dr_y + L
            end
            if dr_z > L_half
                dr_z = dr_z - L
            elseif dr_z < -L_half
                dr_z = dr_z + L
            end
            
            r2 = dr_x*dr_x + dr_y*dr_y + dr_z*dr_z
            
            if r2 < rc2 && r2 > 0.0
                # Use mixed pair routine for consistency with atomic paths
                # For now, use type 1 (single-component default)
                # TODO: Support per-site types from molecule templates
                type_i = 1
                type_j = 1
                energy += lj_pair_u_from_r2_mixed(r2, type_i, type_j, p)
            end
        end
    end
    
    return energy
end

"""
    molecule_translation_trial!(sys::MolecularSystem, mol_idx::Int, p::LJParams, max_disp::Float64)::Bool

Perform translation trial move for molecule mol_idx.
Returns true if accepted, false if rejected.
"""
function molecule_translation_trial!(sys::MolecularSystem, mol_idx::Int, p::LJParams, max_disp::Float64)::Bool
    mol = sys.molecules[mol_idx]
    template = sys.templates[mol.template_idx]
    L = sys.cl.L
    
    # Store old COM
    com_old = copy(mol.com)
    
    # Compute old energy (sum over all sites in molecule)
    Eold = 0.0
    site_start = sys.n_atoms
    for m in 1:(mol_idx-1)
        site_start += sys.templates[sys.molecules[m].template_idx].n_sites
    end
    for site_local_idx in 1:template.n_sites
        site_idx = site_start + site_local_idx
        Eold += molecular_local_energy(site_idx, sys, p)
    end
    
    # Generate trial displacement
    dx = (rand(sys.rng) - 0.5) * 2.0 * max_disp
    dy = (rand(sys.rng) - 0.5) * 2.0 * max_disp
    dz = (rand(sys.rng) - 0.5) * 2.0 * max_disp
    
    # Apply trial move
    mol.com[1] = com_old[1] + dx
    mol.com[2] = com_old[2] + dy
    mol.com[3] = com_old[3] + dz
    
    # Wrap COM
    wrap!(mol.com, L)
    
    # Update site positions
    update_site_positions!(sys)
    
    # Compute new energy
    Enew = 0.0
    for site_local_idx in 1:template.n_sites
        site_idx = site_start + site_local_idx
        Enew += molecular_local_energy(site_idx, sys, p)
    end
    
    # Metropolis acceptance
    ΔE = Enew - Eold
    accepted = false
    if ΔE <= 0.0 || rand(sys.rng) < exp(-p.β * ΔE)
        accepted = true
    else
        # Reject: restore old COM
        copyto!(mol.com, com_old)
        update_site_positions!(sys)
    end
    
    return accepted
end

"""
    molecule_rotation_trial!(sys::MolecularSystem, mol_idx::Int, p::LJParams)::Bool

Perform rotation trial move for molecule mol_idx.
Returns true if accepted, false if rejected.
"""
function molecule_rotation_trial!(sys::MolecularSystem, mol_idx::Int, p::LJParams)::Bool
    mol = sys.molecules[mol_idx]
    template = sys.templates[mol.template_idx]
    L = sys.cl.L
    
    # Store old quaternion
    quat_old = copy(mol.quat)
    
    # Compute old energy
    Eold = 0.0
    site_start = sys.n_atoms
    for m in 1:(mol_idx-1)
        site_start += sys.templates[sys.molecules[m].template_idx].n_sites
    end
    for site_local_idx in 1:template.n_sites
        site_idx = site_start + site_local_idx
        Eold += molecular_local_energy(site_idx, sys, p)
    end
    
    # Generate new random quaternion (uniform on SO(3))
    mol.quat = uniform_random_quaternion(sys.rng)
    
    # Update site positions
    update_site_positions!(sys)
    
    # Compute new energy
    Enew = 0.0
    for site_local_idx in 1:template.n_sites
        site_idx = site_start + site_local_idx
        Enew += molecular_local_energy(site_idx, sys, p)
    end
    
    # Metropolis acceptance (no Jacobian needed for uniform rotation)
    ΔE = Enew - Eold
    accepted = false
    if ΔE <= 0.0 || rand(sys.rng) < exp(-p.β * ΔE)
        accepted = true
    else
        # Reject: restore old quaternion
        copyto!(mol.quat, quat_old)
        update_site_positions!(sys)
    end
    
    return accepted
end

# ============================================================================
# CBMC (Configurational Bias Monte Carlo) for Grand-Canonical μVT
# ============================================================================

"""
    molecule_interaction_energy(sys, template, r_com, q, p; site_pos_buffer)::Float64

Compute interaction energy between a hypothetical molecule (at COM r_com with orientation q)
and ALL existing sites/molecules in the system.
Excludes intramolecular interactions (only intermolecular).
Uses site_pos_buffer to store computed site positions (must be 3 x template.n_sites).

This function does NOT modify the system - it's for trial evaluation only.
"""
function molecule_interaction_energy(
    sys::MolecularSystem,
    template::MoleculeTemplate,
    r_com::AbstractVector{Float64},
    q::AbstractVector{Float64},
    p::LJParams;
    site_pos_buffer::Matrix{Float64}
)::Float64
    L = sys.cl.L
    rc2 = p.rc2
    ncell = sys.cl.ncell
    existing_pos = sys.site_pos
    dr = sys.scratch_dr
    
    # Compute rotation matrix
    R = quaternion_to_rotation_matrix(q)
    
    # Compute candidate molecule site positions in buffer
    n_sites = template.n_sites
    @inbounds for site_local_idx in 1:n_sites
        # Body frame position
        r_body = template.sites_body[:, site_local_idx]
        
        # Rotate
        r_rot = R * r_body
        
        # Add COM and wrap
        x = r_com[1] + r_rot[1]
        y = r_com[2] + r_rot[2]
        z = r_com[3] + r_rot[3]
        
        # Wrap to [0, L)
        x_wrapped = x - L * floor(x / L)
        y_wrapped = y - L * floor(y / L)
        z_wrapped = z - L * floor(z / L)
        
        site_pos_buffer[1, site_local_idx] = x_wrapped
        site_pos_buffer[2, site_local_idx] = y_wrapped
        site_pos_buffer[3, site_local_idx] = z_wrapped
    end
    
    # Compute interaction energy: sum over candidate sites interacting with all existing sites
    # Use the same structure as molecular_total_energy for consistency
    energy = 0.0
    N_existing = sys.n_sites_total
    
    # Loop over all candidate sites and all existing sites (same convention as molecular_total_energy)
    @inbounds for cand_site_idx in 1:n_sites
        cand_x = site_pos_buffer[1, cand_site_idx]
        cand_y = site_pos_buffer[2, cand_site_idx]
        cand_z = site_pos_buffer[3, cand_site_idx]
        
        for j in 1:N_existing
            # Compute distance vector (same convention as molecular_total_energy)
            dr_x = existing_pos[1, j] - cand_x
            dr_y = existing_pos[2, j] - cand_y
            dr_z = existing_pos[3, j] - cand_z
            
            # Apply minimum image convention (same as molecular_total_energy)
            L_half = L / 2.0
            if dr_x > L_half
                dr_x = dr_x - L
            elseif dr_x < -L_half
                dr_x = dr_x + L
            end
            if dr_y > L_half
                dr_y = dr_y - L
            elseif dr_y < -L_half
                dr_y = dr_y + L
            end
            if dr_z > L_half
                dr_z = dr_z - L
            elseif dr_z < -L_half
                dr_z = dr_z + L
            end
            
            r2 = dr_x*dr_x + dr_y*dr_y + dr_z*dr_z
            
            if r2 < rc2 && r2 > 0.0
                # Use mixed pair routine for consistency with atomic paths
                # For now, use type 1 (single-component default)
                # TODO: Support per-site types from molecule templates
                type_i = 1
                type_j = 1
                energy += lj_pair_u_from_r2_mixed(r2, type_i, type_j, p)
            end
        end
    end
    
    return energy
end

"""
    molecule_current_interaction_energy(sys, mol_idx, p)::Float64

Compute interaction energy of existing molecule mol_idx with the rest of the system.
Excludes intramolecular interactions.
"""
function molecule_current_interaction_energy(
    sys::MolecularSystem,
    mol_idx::Int,
    p::LJParams
)::Float64
    template = sys.templates[sys.molecules[mol_idx].template_idx]
    
    # Find site indices for this molecule
    site_start = sys.n_atoms
    for m in 1:(mol_idx-1)
        site_start += sys.templates[sys.molecules[m].template_idx].n_sites
    end
    
    # Sum local energy over all sites in molecule
    energy = 0.0
    for site_local_idx in 1:template.n_sites
        site_idx = site_start + site_local_idx
        energy += molecular_local_energy(site_idx, sys, p)
    end
    
    return energy
end

"""
    categorical_select(weights, rng)::Int

Select index from categorical distribution with given weights.
Returns index j with probability weights[j] / sum(weights).
Allocation-free: uses linear scan.
"""
function categorical_select(weights::Vector{Float64}, rng::Xoshiro)::Int
    # Compute cumulative sum
    total = 0.0
    for w in weights
        total += w
    end
    
    if total <= 0.0
        # All weights zero or negative - select uniformly
        return rand(rng, 1:length(weights))
    end
    
    # Sample uniform [0, total)
    u = rand(rng) * total
    
    # Linear scan to find selected index
    cumsum = 0.0
    for j in 1:length(weights)
        cumsum += weights[j]
        if u < cumsum
            return j
        end
    end
    
    # Fallback (shouldn't happen with exact arithmetic)
    return length(weights)
end

"""
    count_molecules_of_template(sys, template_idx)::Int

Count number of molecules with given template index.
"""
function count_molecules_of_template(sys::MolecularSystem, template_idx::Int)::Int
    count = 0
    for mol in sys.molecules
        if mol.template_idx == template_idx
            count += 1
        end
    end
    return count
end

"""
    CBMCInsertDiagnostics

Accumulator for CBMC insertion diagnostics.
Records per-attempt data for analysis.
"""
mutable struct CBMCInsertDiagnostics
    k_trials::Vector{Int}
    W_ins::Vector{Float64}
    j_star::Vector{Int}
    ΔU_candidates::Vector{Vector{Float64}}  # Vector of ΔU for each candidate
    accept_prob::Vector{Float64}
    accepted::Vector{Bool}
    
    function CBMCInsertDiagnostics()
        return new(
            Int[],
            Float64[],
            Int[],
            Vector{Float64}[],
            Float64[],
            Bool[]
        )
    end
end

"""
    reset!(diag::CBMCInsertDiagnostics)

Reset diagnostics accumulator.
"""
function reset!(diag::CBMCInsertDiagnostics)
    empty!(diag.k_trials)
    empty!(diag.W_ins)
    empty!(diag.j_star)
    empty!(diag.ΔU_candidates)
    empty!(diag.accept_prob)
    empty!(diag.accepted)
    return nothing
end

"""
    CBMCDeleteDiagnostics

Accumulator for CBMC deletion diagnostics.
Records per-attempt data for analysis.
"""
mutable struct CBMCDeleteDiagnostics
    k_trials::Vector{Int}
    W_del::Vector{Float64}
    ΔU_real::Vector{Float64}
    ΔU_decoys::Vector{Vector{Float64}}  # Vector of ΔU for each decoy
    accept_prob::Vector{Float64}
    accepted::Vector{Bool}
    
    function CBMCDeleteDiagnostics()
        return new(
            Int[],
            Float64[],
            Float64[],
            Vector{Float64}[],
            Float64[],
            Bool[]
        )
    end
end

"""
    reset!(diag::CBMCDeleteDiagnostics)

Reset diagnostics accumulator.
"""
function reset!(diag::CBMCDeleteDiagnostics)
    empty!(diag.k_trials)
    empty!(diag.W_del)
    empty!(diag.ΔU_real)
    empty!(diag.ΔU_decoys)
    empty!(diag.accept_prob)
    empty!(diag.accepted)
    return nothing
end

"""
    cbmc_insert_trial!(sys, template_idx, p; beta, z, k_trials=10, rng=nothing, diag=nothing)::Bool

CBMC insertion trial move for grand-canonical μVT ensemble.
Proposes inserting one molecule of template template_idx.

If `diag` is provided (CBMCInsertDiagnostics), records diagnostics for this attempt.

Algorithm:
1. Generate k_trials independent candidates (uniform COM + uniform quaternion)
2. Compute weights w_j = exp(-β*ΔU_j) for each candidate
3. Compute Rosenbluth weight W_ins = Σ_j w_j (sum over all k_trials candidates)
4. Select candidate j* with probability w_j*/W_ins (categorical distribution)
5. Accept with A_ins = min(1, (z*V/(N+1)) * (W_ins/k_trials))
6. If accepted, insert molecule and update system

Note: Uniform proposals (COM and quaternion) have no Jacobian.
The 1/k_trials factor in A_ins maintains detailed balance with deletion moves.

Returns true if accepted, false if rejected.
"""
function cbmc_insert_trial!(
    sys::MolecularSystem,
    template_idx::Int,
    p::LJParams;
    beta::Float64,
    z::Float64,
    k_trials::Int=10,
    rng::Union{Xoshiro, Nothing}=nothing,
    diag::Union{CBMCInsertDiagnostics, Nothing}=nothing
)::Bool
    if rng === nothing
        rng = sys.rng
    end
    
    template = sys.templates[template_idx]
    L = sys.cl.L
    V = L * L * L
    N = count_molecules_of_template(sys, template_idx)
    
    # Allocate buffers for candidate site positions
    site_pos_buffer = zeros(Float64, 3, template.n_sites)
    
    # Generate k_trials candidates and compute weights
    candidates_com = Vector{MVector{3,Float64}}()
    candidates_quat = Vector{MVector{4,Float64}}()
    weights = Vector{Float64}(undef, k_trials)
    
    # Store ΔU for diagnostics
    ΔU_candidates = diag !== nothing ? Vector{Float64}(undef, k_trials) : Float64[]
    
    for j in 1:k_trials
        # Uniform COM in box [0, L)
        r_com = MVector{3,Float64}(rand(rng) * L, rand(rng) * L, rand(rng) * L)
        q = uniform_random_quaternion(rng)
        
        push!(candidates_com, r_com)
        push!(candidates_quat, q)
        
        # Compute interaction energy
        ΔU_j = molecule_interaction_energy(sys, template, r_com, q, p; site_pos_buffer=site_pos_buffer)
        
        # Store for diagnostics
        if diag !== nothing
            ΔU_candidates[j] = ΔU_j
        end
        
        # Compute weight
        weights[j] = exp(-beta * ΔU_j)
    end
    
    # Compute Rosenbluth weight
    W_ins = sum(weights)
    
    # Select candidate with probability proportional to weight
    j_star = categorical_select(weights, rng)
    
    # Record diagnostics if requested
    if diag !== nothing
        push!(diag.k_trials, k_trials)
        push!(diag.W_ins, W_ins)
        push!(diag.j_star, j_star)
        push!(diag.ΔU_candidates, copy(ΔU_candidates))
    end
    
    # Acceptance probability (Frenkel-Smit: includes 1/k factor)
    # A_ins = min(1, (z*V/(N+1)) * (W_ins/k))
    A_ins = (z * V / (N + 1)) * (W_ins / k_trials)
    accept_prob = min(1.0, A_ins)
    
    # Record acceptance probability in diagnostics
    if diag !== nothing
        push!(diag.accept_prob, accept_prob)
    end
    
    # Metropolis acceptance
    accepted = false
    if rand(rng) < accept_prob
        accepted = true
        
        # Insert molecule: add to molecules vector
        new_mol = MoleculeState(
            candidates_com[j_star],
            candidates_quat[j_star],
            template_idx
        )
        push!(sys.molecules, new_mol)
        sys.n_molecules += 1
        
        # Update site mappings and positions
        # Recompute n_sites_total
        old_n_sites_total = sys.n_sites_total
        sys.n_sites_total = sys.n_atoms
        for mol in sys.molecules
            sys.n_sites_total += sys.templates[mol.template_idx].n_sites
        end
        
        # Resize site arrays
        new_site_pos = zeros(Float64, 3, sys.n_sites_total)
        new_site_to_molecule = Vector{Int}(undef, sys.n_sites_total)
        new_site_to_site_idx = Vector{Int}(undef, sys.n_sites_total)
        
        # Copy existing data
        for i in 1:old_n_sites_total
            new_site_pos[1, i] = sys.site_pos[1, i]
            new_site_pos[2, i] = sys.site_pos[2, i]
            new_site_pos[3, i] = sys.site_pos[3, i]
            new_site_to_molecule[i] = sys.site_to_molecule[i]
            new_site_to_site_idx[i] = sys.site_to_site_idx[i]
        end
        
        # Update mappings for new molecule
        site_idx = old_n_sites_total
        for site_local_idx in 1:template.n_sites
            site_idx += 1
            new_site_to_molecule[site_idx] = sys.n_molecules
            new_site_to_site_idx[site_idx] = site_local_idx
        end
        
        sys.site_pos = new_site_pos
        sys.site_to_molecule = new_site_to_molecule
        sys.site_to_site_idx = new_site_to_site_idx
        
        # Recreate cell list with new size
        sys.cl = CellList(sys.n_sites_total, L, sys.cl.rc)
        
        # Update site positions
        update_site_positions!(sys)
        
        # Rebuild cell list
        rebuild_cells_molecular!(sys, L)
        
        # Track acceptance
        sys.cbmc_insert_accepted += 1
    end
    
    # Record acceptance status in diagnostics
    if diag !== nothing
        push!(diag.accepted, accepted)
    end
    
    sys.cbmc_insert_attempted += 1
    return accepted
end

"""
    cbmc_delete_trial!(sys, template_idx, p; beta, z, k_trials=10, rng=nothing, diag=nothing)::Bool

CBMC deletion trial move for grand-canonical μVT ensemble.
Proposes deleting one molecule of template template_idx.

If `diag` is provided (CBMCDeleteDiagnostics), records diagnostics for this attempt.

Algorithm:
1. If N==0, reject
2. Select existing molecule i uniformly (probability 1/N)
3. Compute ΔU_real = interaction energy of molecule i with rest of system
4. Generate k_trials-1 decoy candidates (uniform COM + uniform quaternion)
5. Compute Rosenbluth weight W_del = w_real + Σ_j w_j where:
   - w_real = exp(-β*ΔU_real) for the actual molecule
   - w_j = exp(-β*ΔU_j) for decoy candidates
   - Total k_trials candidates (1 real + k_trials-1 decoys)
6. Accept with A_del = min(1, (N/(z*V)) * (k_trials/W_del))
7. If accepted, delete molecule and update system

Note: Uniform proposals have no Jacobian. The k_trials factor in A_del maintains
detailed balance with insertion moves (symmetric to 1/k_trials in insertion).

Returns true if accepted, false if rejected.
"""
function cbmc_delete_trial!(
    sys::MolecularSystem,
    template_idx::Int,
    p::LJParams;
    beta::Float64,
    z::Float64,
    k_trials::Int=10,
    rng::Union{Xoshiro, Nothing}=nothing,
    diag::Union{CBMCDeleteDiagnostics, Nothing}=nothing
)::Bool
    if rng === nothing
        rng = sys.rng
    end
    
    template = sys.templates[template_idx]
    L = sys.cl.L
    V = L * L * L
    N = count_molecules_of_template(sys, template_idx)
    
    # Reject if no molecules
    if N == 0
        return false
    end
    
    # Find all molecules with this template
    mol_indices = Int[]
    for (idx, mol) in enumerate(sys.molecules)
        if mol.template_idx == template_idx
            push!(mol_indices, idx)
        end
    end
    
    # Select molecule uniformly
    selected_idx_in_list = rand(rng, 1:length(mol_indices))
    mol_idx = mol_indices[selected_idx_in_list]
    selected_mol = sys.molecules[mol_idx]
    
    # Compute real interaction energy
    ΔU_real = molecule_current_interaction_energy(sys, mol_idx, p)
    w_real = exp(-beta * ΔU_real)
    
    # Generate k_trials-1 decoy candidates
    site_pos_buffer = zeros(Float64, 3, template.n_sites)
    decoy_weights = Vector{Float64}(undef, k_trials - 1)
    ΔU_decoys = diag !== nothing ? Vector{Float64}(undef, k_trials - 1) : Float64[]
    
    for j in 1:(k_trials - 1)
        # Uniform COM + uniform quaternion
        r_com = MVector{3,Float64}(rand(rng) * L, rand(rng) * L, rand(rng) * L)
        q = uniform_random_quaternion(rng)
        
        # Compute interaction energy
        ΔU_j = molecule_interaction_energy(sys, template, r_com, q, p; site_pos_buffer=site_pos_buffer)
        
        # Store for diagnostics
        if diag !== nothing
            ΔU_decoys[j] = ΔU_j
        end
        
        # Compute weight
        decoy_weights[j] = exp(-beta * ΔU_j)
    end
    
    # Compute Rosenbluth weight
    W_del = w_real + sum(decoy_weights)
    
    # Record diagnostics if requested
    if diag !== nothing
        push!(diag.k_trials, k_trials)
        push!(diag.W_del, W_del)
        push!(diag.ΔU_real, ΔU_real)
        push!(diag.ΔU_decoys, copy(ΔU_decoys))
    end
    
    # Acceptance probability (Frenkel-Smit: includes k factor)
    # A_del = min(1, (N/(z*V)) * (k/W_del))
    if W_del <= 0.0
        if diag !== nothing
            push!(diag.accept_prob, 0.0)
            push!(diag.accepted, false)
        end
        return false  # Reject if weight is zero or negative
    end
    
    A_del = (N / (z * V)) * (Float64(k_trials) / W_del)
    accept_prob = min(1.0, A_del)
    
    # Record acceptance probability in diagnostics
    if diag !== nothing
        push!(diag.accept_prob, accept_prob)
    end
    
    # Metropolis acceptance
    accepted = false
    if rand(rng) < accept_prob
        accepted = true
        
        # Delete molecule: remove from molecules vector
        deleteat!(sys.molecules, mol_idx)
        sys.n_molecules -= 1
        
        # Recompute n_sites_total
        sys.n_sites_total = sys.n_atoms
        for mol in sys.molecules
            sys.n_sites_total += sys.templates[mol.template_idx].n_sites
        end
        
        # Rebuild site mappings from scratch
        # Atoms map to -1
        new_site_to_molecule = Vector{Int}(undef, sys.n_sites_total)
        new_site_to_site_idx = Vector{Int}(undef, sys.n_sites_total)
        
        for i in 1:sys.n_atoms
            new_site_to_molecule[i] = -1
            new_site_to_site_idx[i] = i
        end
        
        # Molecules map to molecule index (after deletion, indices are shifted)
        site_idx = sys.n_atoms
        for (new_mol_idx, mol) in enumerate(sys.molecules)
            template_local = sys.templates[mol.template_idx]
            for site_local_idx in 1:template_local.n_sites
                site_idx += 1
                new_site_to_molecule[site_idx] = new_mol_idx
                new_site_to_site_idx[site_idx] = site_local_idx
            end
        end
        
        # Resize arrays to new size
        new_site_pos = zeros(Float64, 3, sys.n_sites_total)
        
        sys.site_to_molecule = new_site_to_molecule
        sys.site_to_site_idx = new_site_to_site_idx
        
        # Update site positions
        update_site_positions!(sys)
        
        # Recreate cell list with new size
        sys.cl = CellList(sys.n_sites_total, L, sys.cl.rc)
        
        # Rebuild cell list
        rebuild_cells_molecular!(sys, L)
        
        # Track acceptance
        sys.cbmc_delete_accepted += 1
    end
    
    # Record acceptance status in diagnostics
    if diag !== nothing
        push!(diag.accepted, accepted)
    end
    
    sys.cbmc_delete_attempted += 1
    return accepted
end
