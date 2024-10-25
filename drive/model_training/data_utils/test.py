import numpy as np 



def test():

    list_ = [i for i in range(10)]

    yield list_ 


def compute_square():


    x = np.array(test)

    print(x)
    return 


compute_square()