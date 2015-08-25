import shelve

class ShelfInterface:
	def __init__(self, filename):
		self.shelf = shelve.open(filename)

	def close(self):
		self.shelf.close()

	def insert(self, key, array):
		if isinstance(key, list):
			key = self.list_to_str(key)

		self.shelf[key] = array

	def get(self, key):
		if isinstance(key, list):
			key = self.list_to_str(key)

		return self.shelf[key]

	def list_to_str(self, list_key):
		sep = ''
		return sep.join(str(i) for i in list_key)

	def str_to_list(self, str_key):
		sep = ''
		return [int(s) for s in str_key]
