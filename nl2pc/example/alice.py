import nl2pc
import random
import time
from contextlib import contextmanager

import numpy as np

random.seed(1)


@contextmanager
def timer(text=''):
    """Helper for measuring runtime"""

    time_start = time.perf_counter()
    yield
    print('---{} time: {:.2f} s'.format(text, time.perf_counter()-time_start))


def main():

    address = '127.0.0.1'
    port = 7766
    nthreads = 2

    n = 4194304
    alice = nl2pc.Create(nl2pc.ckks_role.SERVER, address=address, port=port, nthreads=nthreads, verbose=False)

    print("---CKKS cmp---")
    for _ in range(3000):
        x1 = [random.random()-0.5 for _ in range(n)]
        x2 = [random.random()-0.5 for _ in range(n)]
        x = [x1[i] + x2[i] for i in range(n)]
        tres = list()
        for i in range(n):
            m = 1 if x[i] > 0 else 0
            tres.append(m)
        with timer():
            res = alice.cmp(x1)
        print(sum(np.abs(np.array(res)-np.array(tres))))

    # print("---CKKS max---")
    # for _ in range(300):
    #     x1 = [random.random()-0.5 for _ in range(n)]
    #     x2 = [random.random()-0.5 for _ in range(n)]
    #     x = [x1[i] + x2[i] for i in range(n)]
    #     tres = list()
    #     for i in range(int(n/4)):
    #         m = np.argmax(np.array([x[i], x[int(n*0.25)+i], x[int(n*0.5)+i], x[int(n*0.75)+i]]))
    #         tres.append(m)
    #     # x = [1.00000001, 9.0000001, 1., 9.]
    #     with timer():
    #         res = alice.max(x1, 4)
    #     print(sum(np.abs(np.array(res)-np.array(tres))))

    # print("---CKKS relu---")
    # for _ in range(3):
    #     x = [-1*random.random() for _ in range(n)]
    #     with timer():
    #         res = alice.relu(x)
    #     print(x[-3:])
    #     print(res[-3:])

    # print("---CKKS maxpool---")
    # for _ in range(3):
    #     x = [-1*random.random() for _ in range(n)]
    #     # x = [-4.1, 1.03, 4.01, 0.43, 4.2, 12, 3, -12]
    #     with timer():
    #         res = alice.maxpool2d(x, 9)
    #     print(x[24999], x[49999], x[74999], x[99999])
    #     print(res[24999])


if __name__ == "__main__":
    main()