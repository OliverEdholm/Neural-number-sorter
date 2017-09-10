# imports
from config import N_BITS
from config import N_NUMBERS
from config import BATCH_SIZE
from config import N_EPOCHS
from data import get_batch

from six.moves import xrange

import numpy as np
import tensorflow as tf


# functions
def build_model(X):
	input_layer = tf.reshape(X, [BATCH_SIZE, N_NUMBERS * N_BITS])

	dense1 = tf.layers.dense(input_layer, 200, activation=tf.nn.relu)
	dense2 = tf.layers.dense(dense1, 200, activation=tf.nn.relu)
	dense3 = tf.layers.dense(dense2, N_NUMBERS * N_BITS)

	output_layer = tf.reshape(dense3, [BATCH_SIZE, N_NUMBERS, N_BITS])

	return output_layer


def convert_prediction(prediction):
	def find_nearest(value, array=[0, 1]):
	    idx = (np.abs(array-value)).argmin()

	    return array[idx]

	output = []
	for bits in prediction:
		output_bits = []
		for bit in bits:
			output_bit = find_nearest(bit)
			output_bits.append(output_bit)

		output.append(np.array(output_bits))

	return np.array(output)



def main():
	X = tf.placeholder('float', [None, N_NUMBERS, N_BITS])
	y = tf.placeholder('float', [None, N_NUMBERS, N_BITS])

	y_hat = build_model(X)

	loss = tf.losses.mean_squared_error(y, y_hat)

	optimizer = tf.train.AdamOptimizer()
	train_operation = optimizer.minimize(loss)

	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)

		for epoch in xrange(N_EPOCHS):
			x_batch, y_batch = get_batch()

			_, pred, loss_val = sess.run([train_operation, y_hat, loss],
       								feed_dict={X: x_batch, y: y_batch})

			if (epoch + 1) % 2000 == 0:
				print('epoch: {}, loss: {}'.format(epoch + 1, loss_val))
				print('Input: ')
				print(x_batch[0])
				print()
				print('Ground truth:')
				print(y_batch[0])
				print()
				print('Prediction:')
				print(convert_prediction(pred[0]))
				print()
				print()


		
if __name__ == '__main__':
	main()
