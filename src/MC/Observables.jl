"""
Observable computation utilities (energy, virial, pressure).
"""

# Overlap cap (reduced units): if r < r_cap, set U/ε = 1000 and force = 0
const OVERLAP_U_REDUCED = 100
const OVERLAP_R_REDUCED = 0.75                     #0.6278893464830442  # U=1000

using Base.Threads

"""
    lj_pair_u_from_r2(r2, p)::Float64

Compute LJ pair energy from r2, using LJ model specified in p.
BACKWARD-COMPATIBLE: Uses global σ/ϵ for single-component.
For multicomponent, use lj_pair_u_from_r2_mixed instead.
For :truncated: u(r) = 4ε[(σ/r)^12 - (σ/r)^6]
For :shifted: u_shift(r) = u(r) - u(rc)
Returns 0.0 if r2 >= rc2.
"""
@inline function lj_pair_u_from_r2(r2::Float64, p::LJParams)::Float64
    if r2 >= p.rc2 || r2 <= 0.0
        return 0.0
    end
    # Overlap cap: if r < r_cap, return capped energy
    rcap2 = (OVERLAP_R_REDUCED * p.σ)^2
    if r2 < rcap2
        return OVERLAP_U_REDUCED * p.ϵ
    end
    # Guard against very small distances (must not allocate)
    if r2 < 1e-14
        return Inf
    end
    σ2 = p.σ * p.σ
    invr2 = σ2 / r2
    invr6 = invr2 * invr2 * invr2
    u_unshifted = 4.0 * p.ϵ * (invr6 * invr6 - invr6)
    
    if p.lj_model == :shifted
        return u_unshifted - p.u_rc
    else  # :truncated (default)
        return u_unshifted
    end
end

"""
    lj_pair_u_from_r2_mixed(r2, type_i, type_j, p)::Float64

Compute LJ pair energy from r2 using Lorentz-Berthelot mixed parameters.
For type_i and type_j, uses σ_ab = (σ_a + σ_b)/2, ϵ_ab = sqrt(ϵ_a * ϵ_b).
For :truncated: u(r) = 4ε_ab[(σ_ab/r)^12 - (σ_ab/r)^6]
For :shifted: u_shift(r) = u(r) - u_ab(rc) where u_ab(rc) uses mixed parameters.
Returns 0.0 if r2 >= rc2.
"""
@inline function lj_pair_u_from_r2_mixed(r2::Float64, type_i::Int, type_j::Int, p::LJParams)::Float64
    if r2 >= p.rc2 || r2 <= 0.0
        return 0.0
    end
    # Guard against very small distances (must not allocate)
    if r2 < 1e-14
        return Inf
    end
    
    # Get mixed parameters (Lorentz-Berthelot)
    if p.n_types == 1
        # Single-component fallback (backward compatibility)
        σ_ab = p.σ
        ϵ_ab = p.ϵ
    else
        # Multicomponent: use mixing table
        @inbounds σ_ab = p.σ_mix[type_i, type_j]
        @inbounds ϵ_ab = p.ϵ_mix[type_i, type_j]
    end
    
    # Overlap cap: if r < r_cap, return capped energy
    rcap2 = (OVERLAP_R_REDUCED * σ_ab)^2
    if r2 < rcap2
        return OVERLAP_U_REDUCED * ϵ_ab
    end

    σ2_ab = σ_ab * σ_ab
    invr2 = σ2_ab / r2
    invr6 = invr2 * invr2 * invr2
    u_unshifted = 4.0 * ϵ_ab * (invr6 * invr6 - invr6)
    
    if p.lj_model == :shifted
        # Compute u_ab(rc) for this pair type
        inv_rc2 = 1.0 / (p.rc * p.rc)
        σ2_ab_rc2 = σ2_ab * inv_rc2
        invr6_rc = σ2_ab_rc2 * σ2_ab_rc2 * σ2_ab_rc2
        invr12_rc = invr6_rc * invr6_rc
        u_ab_rc = 4.0 * ϵ_ab * (invr12_rc - invr6_rc)
        return u_unshifted - u_ab_rc
    else  # :truncated (default)
        return u_unshifted
    end
end

"""
    total_energy(st, p)::Float64

Compute total energy of the system (O(N^2), can be slower/allocating).
Double-count-safe: only counts pairs i<j.
Uses mixed parameters if multicomponent (type information from st.types).
"""
function total_energy(st, p)::Float64
    energy = (nthreads() > 1 && st.N >= 200) ? _total_pair_energy_threaded(st, p) : total_pair_energy_serial(st, p)
    # Add long-range correction if enabled
    if p.use_lrc
        energy += st.N * p.lrc_u_per_particle
    end
    return energy
end

"""
    total_pair_energy_serial(st, p)::Float64

The LJ pair sum of `total_energy` on the calling thread (no tail); allocation-free. `total_energy` uses it with one
thread or N < 200, and the threaded loop `_total_pair_energy_threaded` otherwise (same pairs, other summation order).
"""
function total_pair_energy_serial(st, p)::Float64
    energy = 0.0
    N = st.N
    L = st.L
    L_half = 0.5 * L
    rc2 = p.rc2
    pos = st.pos
    types = st.types
    use_mixed = p.n_types > 1
    
        @inbounds for i in 1:N
            type_i = types[i]
            for j in (i+1):N
                type_j = types[j]
                
                # Compute distance vector
                dr_x = pos[1, j] - pos[1, i]
                dr_y = pos[2, j] - pos[2, i]
                dr_z = pos[3, j] - pos[3, i]
                
                # Apply minimum image convention
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
                    # Use mixed parameters for multicomponent
                    if use_mixed
                        energy += lj_pair_u_from_r2_mixed(r2, type_i, type_j, p)
                    else
                        # Single-component backward compatibility
                        energy += lj_pair_u_from_r2(r2, p)
                    end
                end
            end
        end
    return energy
end

"Threaded LJ pair sum of `total_energy` (per-thread partial sums; spawning the tasks allocates)."
function _total_pair_energy_threaded(st, p)::Float64
    energy = 0.0
    N = st.N
    L = st.L
    L_half = 0.5 * L
    rc2 = p.rc2
    pos = st.pos
    types = st.types
    use_mixed = p.n_types > 1
    
    sums = zeros(Float64, Base.Threads.maxthreadid())
    @threads for i in 1:N
        tid = threadid()
        type_i = types[i]
        local_sum = 0.0
        for j in (i+1):N
            type_j = types[j]
            
            # Compute distance vector
            dr_x = pos[1, j] - pos[1, i]
            dr_y = pos[2, j] - pos[2, i]
            dr_z = pos[3, j] - pos[3, i]
            
            # Apply minimum image convention
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
                # Use mixed parameters for multicomponent
                if use_mixed
                    local_sum += lj_pair_u_from_r2_mixed(r2, type_i, type_j, p)
                else
                    # Single-component backward compatibility
                    local_sum += lj_pair_u_from_r2(r2, p)
                end
            end
        end
        sums[tid] += local_sum
    end
    energy = sum(sums)
    return energy
end

"""
    energy_bruteforce(st, p)::Float64

Brute-force total energy (O(N^2)) for sanity checks.
Currently identical to `total_energy`, kept as an explicit oracle.
"""
function energy_bruteforce(st, p)::Float64
    return total_energy(st, p)
end

"""
    total_energy_scaled(st, p, scale_factor)::Float64

Compute total energy after uniformly scaling all particle positions and box length
by scale_factor (used for virtual volume change pressure).
"""
function total_energy_scaled(st, p, scale_factor::Float64)::Float64
    energy = 0.0
    N = st.N
    L = st.L
    L_half = 0.5 * L
    rc2 = p.rc2
    pos = st.pos
    types = st.types
    use_mixed = p.n_types > 1

    @inbounds for i in 1:N
        type_i = types[i]
        for j in (i+1):N
            type_j = types[j]

            dr_x = pos[1, j] - pos[1, i]
            dr_y = pos[2, j] - pos[2, i]
            dr_z = pos[3, j] - pos[3, i]

            # Apply minimum image in the original box, then scale
            if dr_x > L_half
                dr_x -= L
            elseif dr_x < -L_half
                dr_x += L
            end
            if dr_y > L_half
                dr_y -= L
            elseif dr_y < -L_half
                dr_y += L
            end
            if dr_z > L_half
                dr_z -= L
            elseif dr_z < -L_half
                dr_z += L
            end

            dr_x *= scale_factor
            dr_y *= scale_factor
            dr_z *= scale_factor
            r2 = dr_x*dr_x + dr_y*dr_y + dr_z*dr_z

            if r2 < rc2 && r2 > 0.0
                if use_mixed
                    energy += lj_pair_u_from_r2_mixed(r2, type_i, type_j, p)
                else
                    energy += lj_pair_u_from_r2(r2, p)
                end
            end
        end
    end

    if p.use_lrc
        V = L * L * L
        ρ = N / V
        ρ_scaled = ρ / (scale_factor * scale_factor * scale_factor)
        energy += N * compute_lrc_energy_per_particle(ρ_scaled, p.rc)
    end

    return energy
end

"""
    virtual_volume_exp_factors(st, p, dlnV)::Tuple{Float64,Float64}

Return ⟨exp(-βΔU)⟩ factors for ±dlnV virtual volume changes.
"""
function virtual_volume_exp_factors(st, p, dlnV::Float64)::Tuple{Float64,Float64}
    scale = exp(dlnV / 3.0)
    U0 = total_energy(st, p)
    U_plus = total_energy_scaled(st, p, scale)
    U_minus = total_energy_scaled(st, p, 1.0 / scale)
    exp_plus = exp(-p.β * (U_plus - U0))
    exp_minus = exp(-p.β * (U_minus - U0))
    return exp_plus, exp_minus
end

"""
    lj_force_magnitude_times_r(r2::Float64, p::LJParams)::Float64

Compute r * f(r) for Lennard-Jones force, where f(r) = -dU/dr.
BACKWARD-COMPATIBLE: Uses global σ/ϵ for single-component.
For multicomponent, use lj_force_magnitude_times_r_mixed instead.
For LJ: U(r) = 4ε[(σ/r)^12 - (σ/r)^6]
For shifted LJ: U_shift(r) = U(r) - U(rc), so f_shift(r) = -dU_shift/dr = -dU/dr = f(r)
Thus the force (and virial) is the same for both models.
Force: f(r) = -dU/dr = 4ε[12σ^12/r^13 - 6σ^6/r^7]
r * f(r) = 4ε[12σ^12/r^12 - 6σ^6/r^6] = r_ij · f_ij (virial contribution)
Returns 0.0 if r2 >= rc2.
Note: This function does not depend on lj_model since force is independent of constant shift.
"""
@inline function lj_force_magnitude_times_r(r2::Float64, p::LJParams)::Float64
    if r2 >= p.rc2 || r2 <= 0.0
        return 0.0
    end
    # Overlap cap: constant potential => zero force
    rcap2 = (OVERLAP_R_REDUCED * p.σ)^2
    if r2 < rcap2
        return 0.0
    end
    # Use invr2 for numerical stability
    invr2 = 1.0 / r2
    σ2 = p.σ * p.σ
    σ2_invr2 = σ2 * invr2
    σ6_invr6 = σ2_invr2 * σ2_invr2 * σ2_invr2
    σ12_invr12 = σ6_invr6 * σ6_invr6
    # f_over_r = 4ε[12σ^12/r^12 - 6σ^6/r^6] / r
    # But we want r * f(r), so multiply by r^2 to get r^2 * f_over_r = r * f(r)
    # Actually: r * f(r) = 4ε[12σ^12/r^12 - 6σ^6/r^6]
    # Since σ^12/r^12 = (σ^2/r^2)^6 = (σ^2*invr2)^6 = σ12_invr12
    # Note: Force is the same for shifted and truncated LJ (shift is constant)
    return 4.0 * p.ϵ * (12.0 * σ12_invr12 - 6.0 * σ6_invr6)
end

"""
    lj_force_magnitude_times_r_mixed(r2::Float64, type_i::Int, type_j::Int, p::LJParams)::Float64

Compute r * f(r) for Lennard-Jones force using Lorentz-Berthelot mixed parameters.
For type_i and type_j, uses σ_ab = (σ_a + σ_b)/2, ϵ_ab = sqrt(ϵ_a * ϵ_b).
Force: f(r) = -dU/dr = 4ε_ab[12σ_ab^12/r^13 - 6σ_ab^6/r^7]
r * f(r) = 4ε_ab[12σ_ab^12/r^12 - 6σ_ab^6/r^6] = r_ij · f_ij (virial contribution)
Returns 0.0 if r2 >= rc2.
Note: Force is independent of lj_model (shift is constant).
"""
@inline function lj_force_magnitude_times_r_mixed(r2::Float64, type_i::Int, type_j::Int, p::LJParams)::Float64
    if r2 >= p.rc2 || r2 <= 0.0
        return 0.0
    end
    
    # Get mixed parameters (Lorentz-Berthelot)
    if p.n_types == 1
        # Single-component fallback (backward compatibility)
        σ_ab = p.σ
        ϵ_ab = p.ϵ
    else
        # Multicomponent: use mixing table
        @inbounds σ_ab = p.σ_mix[type_i, type_j]
        @inbounds ϵ_ab = p.ϵ_mix[type_i, type_j]
    end
    
    # Overlap cap: constant potential => zero force
    rcap2 = (OVERLAP_R_REDUCED * σ_ab)^2
    if r2 < rcap2
        return 0.0
    end

    # Use invr2 for numerical stability
    invr2 = 1.0 / r2
    σ2_ab = σ_ab * σ_ab
    σ2_ab_invr2 = σ2_ab * invr2
    σ6_ab_invr6 = σ2_ab_invr2 * σ2_ab_invr2 * σ2_ab_invr2
    σ12_ab_invr12 = σ6_ab_invr6 * σ6_ab_invr6
    
    return 4.0 * ϵ_ab * (12.0 * σ12_ab_invr12 - 6.0 * σ6_ab_invr6)
end

"""
    total_virial(st, p)::Float64

Compute total virial W = Σ r_ij · f_ij for pairs i<j.
Uses minimum image convention.
Double-count-safe: only counts pairs i<j.
Uses mixed parameters if multicomponent (type information from st.types).
"""
function total_virial(st, p)::Float64
    virial = 0.0
    N = st.N
    L = st.L
    rc2 = p.rc2
    pos = st.pos
    types = st.types
    
    @inbounds for i in 1:N
        type_i = types[i]
        for j in (i+1):N
            type_j = types[j]
            
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
                # Use mixed parameters for multicomponent
                if p.n_types > 1
                    virial += lj_force_magnitude_times_r_mixed(r2, type_i, type_j, p)
                else
                    # Single-component backward compatibility
                    virial += lj_force_magnitude_times_r(r2, p)
                end
            end
        end
    end
    
    return virial
end

"""
    compute_g_rc(st, p; bin_width::Float64=0.02)::Float64

Estimate radial distribution function g(rc) by counting pairs near rc.
Reporting-only helper for impulsive correction.
"""
function compute_g_rc(st, p; bin_width::Float64=0.02)::Float64
    N = st.N
    L = st.L
    rc = p.rc
    rc2 = p.rc2
    
    # Bin around rc: [rc - bin_width, rc + bin_width]
    r_min = rc - bin_width
    r_max = rc + bin_width
    r_min2 = r_min * r_min
    r_max2 = r_max * r_max
    
    # Count pairs in this shell
    pair_count = 0
    L_half = L / 2.0
    pos = st.pos
    
    @inbounds for i in 1:N
        for j in (i+1):N
            dr_x = pos[1, j] - pos[1, i]
            dr_y = pos[2, j] - pos[2, i]
            dr_z = pos[3, j] - pos[3, i]
            
            # Minimum image
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
            
            if r2 >= r_min2 && r2 < r_max2 && r2 > 0.0
                pair_count += 1
            end
        end
    end
    
    # Volume of shell: (4π/3) * (r_max^3 - r_min^3)
    V_shell = (4.0 * π / 3.0) * (r_max^3 - r_min^3)
    
    # Ideal gas pair density at this distance
    ρ = N / (L * L * L)
    n_ideal = 0.5 * N * ρ * V_shell  # 0.5 to avoid double counting
    
    if n_ideal > 0.0
        return Float64(pair_count) / n_ideal
    else
        return 1.0  # Default to ideal gas if no pairs expected
    end
end

"""
    pressure(st, p, T::Float64)::Float64

Compute pressure from virial equation: P = ρ*T + (1/(3V))*W
where W = Σ r_ij · f_ij is the virial.

For both truncated and shifted LJ, if impulsive correction is enabled (reporting only),
adds ΔP_imp = -(2π/3) * ρ² * rc³ * u(rc) * g(rc).
Note: Even for shifted LJ, the force is discontinuous at rc, so impulsive correction
may be needed for thermodynamic consistency with FEP pressure.
"""
function pressure(st, p, T::Float64)::Float64
    N = st.N
    L = st.L
    V = L * L * L
    ρ = N / V
    W = total_virial(st, p)
    P_sampled = ρ * T + W / (3.0 * V)
    
    # Add long-range correction if enabled
    if p.use_lrc
        P_sampled += p.lrc_p
    end
    
    # Add impulsive correction if enabled (reporting only)
    # Note: This correction is reporting-only and does not affect the Hamiltonian
    # Applies to both truncated and shifted LJ because the force is discontinuous at rc
    # (even though shifted LJ has continuous potential, the force derivative is still discontinuous)
    if p.apply_impulsive_correction
        g_rc = compute_g_rc(st, p)
        rc = p.rc
        # For both truncated and shifted, u(rc) is the unshifted value at rc
        # Compute u_unshifted(rc) = u_rc (which is stored in params)
        u_rc_unshifted = p.u_rc
        ΔP_imp = -(2.0 * π / 3.0) * ρ * ρ * rc * rc * rc * u_rc_unshifted * g_rc
        P_sampled += ΔP_imp
    end
    
    return P_sampled
end

"""
    pressure_bruteforce(st, p, T::Float64)::Float64

Brute-force pressure calculation (O(N^2)) for sanity checks.
Uses the same corrections as `pressure`.
"""
function pressure_bruteforce(st, p, T::Float64)::Float64
    return pressure(st, p, T)
end

"""
    pressure_virial_corrected(st, p, T::Float64)::Float64

Compute virial pressure with impulsive correction explicitly labeled.
Useful for reporting when correction is applied.
Returns (P_virial, P_corrected) tuple.
"""
function pressure_virial_corrected(st, p, T::Float64)::Tuple{Float64, Float64}
    N = st.N
    L = st.L
    V = L * L * L
    ρ = N / V
    W = total_virial(st, p)
    P_virial = ρ * T + W / (3.0 * V)
    
    if p.use_lrc
        P_virial += p.lrc_p
    end
    
    if p.apply_impulsive_correction && p.lj_model == :truncated
        g_rc = compute_g_rc(st, p)
        rc = p.rc
        # For truncated LJ, u(rc) is the unshifted value at rc
        u_rc_unshifted = p.u_rc
        ΔP_imp = -(2.0 * π / 3.0) * ρ * ρ * rc * rc * rc * u_rc_unshifted * g_rc
        P_corrected = P_virial + ΔP_imp
        return (P_virial, P_corrected)
    else
        return (P_virial, P_virial)
    end
end
