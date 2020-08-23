import numpy as N
import matplotlib.pyplot as plt

def f(x):
	return x

res = []
qsum = [0]
qsum2 = [0]
avgs = [0]
ICs = []
ICs2 = []

sample_size = 100.
precision = 0.01
max_iterations = 1000

plt.figure()


for i in xrange(max_iterations):
	xs = N.random.uniform(size=sample_size)
	est_f = N.sum(f(xs))/sample_size
	res.append(est_f)

	qsum.append(qsum[-1]+float(i)/(i+1.)*(est_f-avgs[-1])**2.)

	avgs.append((avgs[-1]*i+est_f)/(i+1.))

	ICs.append(3./N.sqrt(float(i+1))*N.sqrt(qsum[-1]/float(i)))
	plt.plot([i,i], [avgs[-1]-ICs[-1], avgs[-1]+ICs[-1]], color='g')

plt.plot(xrange(max_iterations), avgs[1:], color='k')
plt.hlines(y=0.5, xmin=0, xmax=i, zorder=100)
#plt.ylim(ymin=0, ymax=1)

plt.savefig(open('/home/charles/Documents/Tracer/emissive_losses/MC_test_fig.png','w'), dpi=1000)

