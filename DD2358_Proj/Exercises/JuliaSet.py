"""Julia set generator without optional PIL-based image drawing"""
import time
from functools import wraps
import numpy as np
import line_profiler
import atexit
#profile = line_profiler.LineProfiler()
#atexit.register(profile.print_stats)
#from memory_profiler import profile

# area of complex space to investigate
x1, x2, y1, y2 = -1.8, 1.8, -1.8, 1.8
c_real, c_imag = -0.62772, -.42193

# decorator to time
def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        numTries = 10
        times = np.zeros(numTries)
        for i in range(numTries):
            t1 = time.time()
            result = fn(*args, **kwargs)
            t2 = time.time()
            times[i] = t2-t1
        print(f"@timefn: {fn.__name__} took an average of {np.mean(times):.4e} seconds, with a standard deviation of {np.std(times):.4e} s.")
        return result
    return measure_time

#@timefn
#@profile
def calc_pure_python(desired_width, max_iterations, profiler = None):
    """Create a list of complex coordinates (zs) and complex parameters (cs),
    build Julia set"""
    x_step = (x2 - x1) / desired_width
    y_step = (y1 - y2) / desired_width
    x = []
    y = []
    ycoord = y2
    while ycoord > y1:
        y.append(ycoord)
        ycoord += y_step
    xcoord = x1
    while xcoord < x2:
        x.append(xcoord)
        xcoord += x_step
    # build a list of coordinates and the initial condition for each cell.
    # Note that our initial condition is a constant and could easily be removed,
    # we use it to simulate a real-world scenario with several inputs to our
    # function
    zs = []
    cs = []
    for ycoord in y:
        for xcoord in x:
            zs.append(complex(xcoord, ycoord))
            cs.append(complex(c_real, c_imag))

    #print("Length of x:", len(x))
    #print("Total elements:", len(zs))
    start_time = time.time()
    #profiler.enable()
    output = calculate_z_serial_purepython(max_iterations, zs, cs)
    #profiler.disable()
    end_time = time.time()
    secs = end_time - start_time
    #print(calculate_z_serial_purepython.__name__ + " took", secs, "seconds")

    # This sum is expected for a 1000^2 grid with 300 iterations
    # It ensures that our code evolves exactly as we'd intended
    #assert sum(output) == 33219980
    return output
    
#@timefn
#@profile
def calculate_z_serial_purepython(maxiter, zs, cs):
    """Calculate output list using Julia update rule"""
    output = [0] * len(zs)
    for i in range(len(zs)):
        n = 0
        z = zs[i]
        c = cs[i]
        while abs(z) < 2 and n < maxiter:
            z = z * z + c
            n += 1
        output[i] = n
    return output

if __name__ == "__main__":
    # Calculate the Julia set using a pure Python solution with
    # reasonable defaults for a laptop
    calc_pure_python(desired_width=10000, max_iterations=300) 
