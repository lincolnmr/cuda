from numba import cuda
import numpy
from numpy import random

@cuda.jit
def my_kernel(io_array, o_array):
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    
    # Compute flattened index inside the array
    pos = tx + ty * bw

    num = 0
    count = 0  
    i = 0
    while( i < io_array.size):
      num = io_array[pos]
      for n in io_array:
        if num == n:
          count += 1
      o_array[num] = count
      count = 0
      i += 1

    #if pos < io_array.size:  # Check array boundaries [0, 1] pos = 1
    #  io_array[pos] *= 2   # do the computation

# vetor com números aleatórios
data = random.randint(10, size=(10))

# vetor vazio
o_array = numpy.empty([10])

# números de threads por bloco
threads_per_block = 32

# número de blocos por grid
blocks_per_grid = ( data.size + (threads_per_block - 1) )

# iniciando o kernel
my_kernel[blocks_per_grid, threads_per_block](data, o_array )  # esta linha esta apresentando erro, preciso perguntar para você na aula.

# mostra o resultado
print(data)
print(o_array)
