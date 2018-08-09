import numpy as np


def randomContinuous(start,end):
    num = np.random.random()
    return (num * (end - start)) + start

if __name__ == "__main__":
    times = 1000000
    mu1=-1
    mu2=1

    def KMu(sigma, a):  # Gaussian Kernel with sigma = 1  TODO normalize
        return np.exp((-.5) * ((a-mu1) ** 2) / (sigma ** 2)) / ((2 * np.pi * (sigma ** 2)) ** .5) + np.exp((-.5) * ((a-mu2) ** 2) / (sigma ** 2)) / ((2 * np.pi * (sigma ** 2)) ** .5)

    sigma = [.1,.2,.3,.4,.5,.6,.7,.8,.9,1,2,3,4,5,6,7,8,9]
    testBound = 3

    np.random.seed()
    for _ in range(times):
        a = randomContinuous(-testBound, testBound)
        b = randomContinuous(-testBound, testBound)
        firstResult = KMu(sigma[0], a) < KMu(sigma[0], b)
        for s in sigma[1:]:
            if firstResult == (KMu(s, a) < KMu(s, b)):
                print("PASS: " + str(a) + " , " + str(b) + " , " + str(KMu(s, a)) + " , " + str(KMu(s, b)))
            else:
                print("FAIL:")
                print("a:" + str(a) + "\nb:" + str(b) + "\nsigma:" + str(s))
                print("KMu(" + str(s) +",a)=" + str(KMu(s,a)))
                print("KMu(" + str(s) + ",b)=" + str(KMu(s, b)))
                print("\n\nsigma[0]=" + str(sigma[0]))
                print("KMu(" + str(sigma[0]) + ",a)=" + str(KMu(sigma[0], a)))
                print("KMu(" + str(sigma[0]) + ",b)=" + str(KMu(sigma[0], b)))
                raise Exception("asdflkj")