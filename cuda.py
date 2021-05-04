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
    if pos < io_array.size:
      num = io_array[pos]
      for n in io_array:
        if num == n:
          count += 1
      o_array[num] = count
      count = 0

# vetor com números aleatórios
data = random.randint(10, size=(2**16))

# vetor vazio
o_array = numpy.zeros([10]) #inicia um array zerado, para não pegar lixos de memória

# números de threads por bloco
threads_per_block = 32

# número de blocos por grid
blocks_per_grid = ( data.size + (threads_per_block - 1) )

# iniciando o kernel
my_kernel[blocks_per_grid, threads_per_block](data, o_array )

# mostra o resultado
print("VETOR INICIAL")
print(data)
print("VETOR FINAL COM NÚMEROS SOMADOS")
print(o_array)

'''
i = 0
soma = 0
while( i < o_array.size):
  soma = soma + o_array[i]
  print( i, ":" , o_array[i] )
  i= i + 1
print( "T :", soma )
'''
