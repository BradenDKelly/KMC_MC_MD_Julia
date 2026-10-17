"""
CUDA backend of the reciprocal-space Ewald products (DESIGN §13.3 E2). Loaded automatically by `using CUDA` next to
`using MolSim`; `set_recip_backend!(sys.recip, :cuda)` then keeps the committed structure factors, the weights and the
solvent-candidate structure factors on the GPU, and the per-event products run with CUBLAS.
"""
module MolSimCUDAExt

using MolSim
using CUDA

MolSim.MC.backend_array(::Val{:cuda}, A::AbstractArray) = CuArray(A)

end
