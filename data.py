# imports
from config import N_BITS
from config import N_NUMBERS
from config import BATCH_SIZE

from random import randint
from six.moves import xrange

import numpy as np


# classes
class Number:
	def __init__(self, n_bits=N_BITS):
		self.n_bits = n_bits

		self.bits = self._get_random_bits(self.n_bits)

	def _rand_bit(self):
		return randint(0, 1)

	def _get_random_bits(self, n_bits):
		bits = [self._rand_bit() for _ in xrange(n_bits)]

		return np.array(bits)

	def get_num_value(self):
		bit_values = [2 ** idx * bit
					  for idx, bit in enumerate(self.bits)]

		return sum(bit_values) 

	def get_value(self):
		return self.bits

	def __str__(self):
		return 'Number<bits={}, value={}>'.format(self.bits, self.get_num_value())

	def __repr__(self):
		return str(self.get_num_value())


class Series:
	def __init__(self, numbers):
		self.numbers = np.array(numbers)

	def sort(self):
		sorted_numbers = sorted(self.numbers,
	  						    key=lambda n: n.get_num_value())

		return Series(sorted_numbers)

	def get_value(self):
		numbers_in_bits = [number.get_value()
						   for number in self.numbers]

		return np.array(numbers_in_bits)

	def __str__(self):
		return 'Series<{}>'.format([str(number) for number in self.numbers])

	def __repr__(self):
		return str(self.numbers)


class NumberSeries:
	def __init__(self, n_numbers=N_NUMBERS):
		self.numbers = Series(self._get_random_numbers(n_numbers))
		self.sorted_numbers = self.numbers.sort()

	def _get_random_numbers(self, n_numbers):
		numbers = [Number()
				   for _ in xrange(n_numbers)]

		return numbers

	def __str__(self):
		return 'NumberSeries<numbers={}, sorted={}>'.format(str(self.numbers),
														    str(self.sorted_numbers))

	def __repr__(self):
		return '({}, {})'.format(self.numbers, self.sorted_numbers)


# functions
def get_data():
	number_series = NumberSeries()

	X = number_series.numbers.get_value()
	y = number_series.sorted_numbers.get_value()

	return X, y


def get_batch():
	X_batch = []
	y_batch = []

	for _ in xrange(BATCH_SIZE):
		X, y = get_data()

		X_batch.append(X)
		y_batch.append(y)

	return np.array(X_batch), np.array(y_batch)
