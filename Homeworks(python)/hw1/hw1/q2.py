import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import norm

""" set the mean and standard deviation for two gaussian distribution
	and the probability of each class """
mu1=5.0
mu2=15.0
std1, std2=5.0, 5.0
pw1=0.5

# Define a function to calulate separating surface
def separationsurface(mu1, mu2, std1, std2, pw1):
	# define coeficients of the equation
	pw2 = 1 - pw1
	a = 1/(2*std2**2) - 1/(2*std1**2)
	b = mu1/std1**2 - mu2/std2**2
	c = mu2**2/(2*std2**2) - mu1**2/(2*std1**2) + np.log(std2/std1) + np.log(pw1/pw2)
	x = np.roots([a,b,c]) # finds the roots of the equation
	return x

# set the range of x for plot
x_axis = np.linspace(-20,40,100)

# find the separating surface
x = separationsurface(mu1, mu2, std1, std2, pw1) # enter pw1=0.8 to get separating surface
plt.plot(x_axis,norm.pdf(x_axis,mu1,std1))
plt.plot(x_axis,norm.pdf(x_axis,mu2,std2))
plt.xlabel('x')
plt.ylabel('$p(x|w_i)$')
plt.legend(['$p(x|w_1)$','$p(x|w_2)$'])
plt.axvline(x=x, ymin=0, ymax=1,color='k', linestyle='--',linewidth=0.5)
plt.plot(x, norm.pdf(x,mu1,std1), 'o') 
plt.ylim(ymin=0)
plt.savefig('pdfs1.png')
plt.show()
np.random.seed(42)
# generate random datasets from a two gaussian random variables when p(w1)=p(w2)
x_w1=np.random.normal(loc=mu1, scale=std1, size=200)
x_w2=np.random.normal(loc=mu2, scale=std2, size=200)

# find the mean and standard deviation of generated data
m1 = np.mean(x_w1)
s1 = np.std(x_w1)

m2 = np.mean(x_w2)
s2 = np.std(x_w2)
# generate random datasets from a two gaussian random variables when p(w1)=0.80
x_w1_08=np.random.normal(loc=mu1, scale=std1, size=320)
x_w2_02=np.random.normal(loc=mu2, scale=std2, size=80)

# find the mean and standard deviation of generated data
m1_08 = np.mean(x_w1_08)
s1_08 = np.std(x_w1_08)

m2_02 = np.mean(x_w2_02)
s2_02 = np.std(x_w2_02)

x = separationsurface(m1, m2, s1, s2, pw1) # separating point for histogram when p(w1)=p(w2)
# ploting the histograms p(w1)=p(w2)
plt.figure()
plt.hist(x_w1, bins='auto')
plt.hist(x_w2, bins='auto', alpha=0.4, color='r')
plt.plot(x[1],[0],'ro')
# set the position for x value
xy=(x[1]-1,1)
z='x={0:.4f}'.format(x[1])
plt.annotate('{}'.format(z), xy=xy)
plt.title('Histogram of samples, $p(w_1)=p(w_2)$.')
plt.xlabel('x')
plt.ylabel('number of samples in datasets')
plt.savefig('hist05.png')
plt.show()

x = separationsurface(m1_08, m2_02, s1_08, s2_02, pw1=0.8) # separating point for histogram when p(w1)=0.8
# ploting the histograms p(w1)=0.8
plt.figure()
plt.hist(x_w1_08, bins='auto')
plt.hist(x_w2_02, bins='auto', alpha=0.4, color='r')
plt.plot(x[1],[0],'ro')
# set the position for x value
xy=(x[1]-1,1)
z='x={0:.4f}'.format(x[1])
plt.annotate('{}'.format(z), xy=xy)
plt.title('Histogram of samples, $p(w_1)=0.80, p(w_2)=0.20$.')
plt.xlabel('x')
plt.ylabel('number of samples in datasets')
plt.savefig('hist08.png')
plt.show()

