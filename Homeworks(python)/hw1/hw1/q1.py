import numpy as np 
import matplotlib.pyplot as plt 


""" Q1a) set maximum and minimum values for a univariate random variable x 
(a uniformly distributed rv).
 """
low = 10 - np.sqrt(3.0)/2.0
high = 10 + np.sqrt(3.0)/2.0

""" draw N=10 samples of x from the distribution, 
compute the mean and plot the histogram of the means for 500 different experiments.
"""
# define an empty list for means of 500 different experiments.
means_of_10_samples_uniform = []
np.random.seed(42)
# conduct 500 different experiments of drawing 10 samples of x
for experiment in range(500):

	# draw 10 samples
	s10 = np.random.uniform(low=low, high=high, size = 10)

	# compute the mean of samples and append the mean into list of means
	means_of_10_samples_uniform.append(np.mean(s10))


# define an empty list for means of 500 different experiments.
means_of_100_samples_uniform = []

# conduct 500 different experiments of drawing 100 samples of x
for experiment in range(500):

	# draw 10 samples
	s100 = np.random.uniform(low=low, high=high, size = 100)

	# compute the mean of samples and append the mean into list of means
	means_of_100_samples_uniform.append(np.mean(s100))


# Q1b)
# set mean and Standard deviation for Normal distribution
mean = 10
std = 1

# define an empty list for means of 10 samples from 500 experiments
means_of_10_samples_normal = []

# conduct 500 different experiments of drawing 10 samples of normal distribution x
for experiment in range(500):

	# draw 10 samples
	s10 = np.random.normal(loc=mean, scale=std, size=10)
	# compute the mean of samples and append the mean into list of means
	means_of_10_samples_normal.append(np.mean(s10))


# define an empty list for means of 100 samples from 500 experiments
means_of_100_samples_normal = []

# conduct 500 different experiments of drawing 100 samples of normal distribution x
for experiment in range(500):

	# draw 10 samples
	s100 = np.random.normal(loc=mean, scale=std, size=100)
	# compute the mean of samples and append the mean into list of means
	means_of_100_samples_normal.append(np.mean(s100))


plt.figure(figsize=(12,12))
axis = plt.subplot(221)
plt.hist(means_of_10_samples_uniform, bins='auto')
plt.xlabel('samples')
plt.ylabel('number of samples')
plt.title('Histogram of the means for 10 samples of uniform rv')

plt.subplot(222, sharex=axis, sharey=axis)
plt.hist(means_of_100_samples_uniform, bins='auto')
plt.xlabel('samples')
plt.ylabel('number of samples')
plt.title('Histogram of the means for 100 samples of uniform rv')

plt.subplot(223, sharex=axis, sharey=axis)
plt.hist(means_of_10_samples_normal, bins='auto', color='red')
plt.xlabel('samples')
plt.ylabel('number of samples')
plt.title('Histogram of the means for 10 samples of normal rv')

plt.subplot(224, sharex=axis, sharey=axis)
plt.hist(means_of_100_samples_normal, bins='auto', color='red')
plt.xlabel('samples')
plt.ylabel('number of samples')
plt.title('Histogram of the means for 100 samples of normal rv')

plt.savefig('hists.png')
plt.show()






