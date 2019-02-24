import numpy, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

inputs = []
targets = []
random.seed(1234)
numpy.random.seed(1234)

classA = numpy.concatenate(
    (numpy.random.randn(10, 2) * 0.2 + [1.5, 0.0],
     numpy.random.randn(10, 2) * 0.2 + [-1.5, 0.0]))

classB = numpy.random.randn(20, 2) * 0.2 + [0.0, -0.5]

inputs = numpy.concatenate((classA, classB))

targets = numpy.concatenate(
    (numpy.ones(classA.shape[0]),
     -numpy.ones(classB.shape[0])))

N = inputs.shape[0]  # Number of rows ( samples )

permute = list(range(N))

random.shuffle(permute)

inputs = inputs[permute, :]

targets = targets[permute]

start = numpy.zeros(N)

C = 10

B = [(0, C) for b in range(N)]


def zerofun(a):
    res = 0
    for i in range(len(inputs)):
        res += a[i] * targets[i]
    return res


XC = {'type': 'eq', 'fun': zerofun}


def linkern(a, b):
    return numpy.dot(a, b)


def polykern(a, b, p):
    return math.pow(numpy.dot(a, b) + 1, p)


def rbfkern(a, b, sigma):
    return math.exp(-math.pow(numpy.linalg.norm(a-b),2)/ (2 * sigma * sigma))



# def matrix():
#    for i in range(0,len(inputs)):
#        for j in range(0,len(inputs)):
#            kernmatrix[i][j] = (targets[i]*targets[j]*linkern(inputs[i], inputs[j]))
# matrix()

def objective(a):
    res = 0
    asum = 0
    for i in range(0, len(inputs)):
        asum += a[i]
        for j in range(0, len(inputs)):
            res += (a[i] * a[j] * targets[i] * targets[j] * rbfkern(inputs[i], inputs[j], 1.5))
    res = 0.5 * res - asum
    return (res)


ret = minimize(objective, start, bounds=B, constraints=XC)
alpha = ret['x']

print("Is successful? " + str(ret['success']))
print(ret)
alphares = []
inputsres = []
targetsres = []
for i in range(0, len(alpha)):
    if (math.fabs(alpha[i]) > 10e-5):
        alphares.append(alpha[i])
        inputsres.append(inputs[i])
        targetsres.append(targets[i])

b_sum = 0
t_sum = 0
for i in range(0, len(alpha)):
    b_sum += alpha[i] * targets[i] * rbfkern(inputsres[0], inputs[i], 1.5)
b = b_sum - targetsres[0]

def indicator(x, y):
    ind = 0
    for i in range(0, len(alpha)):
        ind += alpha[i] * targets[i] * rbfkern((x, y), inputs[i], 1.5)
    ind = ind - b
    return ind


xgrid = numpy.linspace(-5, 5)
ygrid = numpy.linspace(-4, 4)
grid = numpy.array([[indicator(x, y) for x in xgrid] for y in ygrid])
plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))

plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')
plt.axis('equal')  # Force same scale on both axes
plt.savefig('svmplot.pdf')  # Save a copy in a file
plt.show()  # Show the p l o t on the screen
