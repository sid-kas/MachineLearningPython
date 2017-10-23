import numpy as np
from multiprocessing import pool
import multiprocessing as mp

k = list(map(lambda x:x**2, range(10)))

def job(x):
    return x**2

def gf():
    # if __name__ == '__main__':
    p = pool.Pool(processes =20)
    data = p.map(job, range(10))
    p.close()
    print(data)


if __name__ == '__main__':
    for i in range(50):
        p = mp.Process(target=gf)
        p.start()
        # p.join()