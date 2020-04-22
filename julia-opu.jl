import Pkg
Pkg.add("PyCall")

using PyCall

# numpy and lightonopu need be installed in Python
lgopu = pyimport("lightonopu")
np = pyimport("numpy")

# return a Julia Array
x = np.random.randint(0, 2, size=(3000, 912*1140), dtype=np.uint8)

# initialize OPU object
opu = lgopu.OPU(n_components=10000)

# perform the optical processing
y = opu.transform1d(x)

# alternative
y = @pycall opu.transform1d(x::Array)::PyArray

# check output properties
println(typeof(y))
println(ndims(y))
println(minimum(y), " ", maximum(y))
