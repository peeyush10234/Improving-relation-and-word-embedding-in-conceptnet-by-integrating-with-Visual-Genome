import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_tsne(file_name, fig):

	plt.figure(fig)
	plt.ion()
	object_embeddings = np.load(file_name)
	data = TSNE(n_components=2).fit_transform(object_embeddings)
	labels = json.load(open('objects.json'))

	print(data[28])
	print(data[29])

	plt.subplots_adjust(bottom = 0.1)
	plt.scatter(
	    data[:, 0], data[:, 1], marker='o',
	    cmap=plt.get_cmap('Spectral'))

	for label, x, y in zip(labels, data[:, 0], data[:, 1]):
	    plt.annotate(
	        label,
	        xy=(x, y), xytext=(-10, 10),
	        textcoords='offset points', ha='right', va='bottom',
	        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
	        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

	plt.savefig(file_name.split('.')[0] + 'png')
	plt.show()

if __name__ == '__main__':
	plot_tsne('blended_object_embeddings.npy', 0)
	plot_tsne('cnet_object_embeddings.npy', 1)
	input()
