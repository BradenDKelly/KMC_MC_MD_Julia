"""
CUDA backend of the reciprocal-space Ewald products (DESIGN §13.3 E2). Loaded automatically by `using CUDA` next to
`using MolSim`; `set_recip_backend!(sys.recip, :cuda)` then keeps the committed structure factors, the weights and the
solvent-candidate structure factors on the GPU, and the per-event products run with CUBLAS. `DevicePairs(sys, ff;
backend=:cuda)` runs the real-space pair kernel (DESIGN §15) on the GPU.
"""
module MolSimCUDAExt

using MolSim
using CUDA

MolSim.MC.backend_array(::Val{:cuda}, A::AbstractArray) = CuArray(A)
# device pair kernel (DevicePairs.jl, DESIGN §15): KernelAbstractions backend
MolSim.MC.device_backend(::Val{:cuda}) = CUDABackend()

end
