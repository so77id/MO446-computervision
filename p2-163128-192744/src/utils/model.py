
import numpy as np

class Model:

	def __init__(self, n_min, dim_in, dim_out):

		self.n_min = n_min
		self.dim_in = dim_in
		self.dim_out = dim_out

	def fit(self, x, y):
		assert(x.shape[0] >= self.n_min)
		assert(x.shape[0] == y.shape[0])
		assert(x.shape[1] == self.dim_in)
		assert(y.shape[1] == self.dim_out)
		return self.fit_model(x, y)

	def fit_model(self, x, y):
		raise Exception("'fit_model' not implemented yet!")

	def predict(self, x):
		assert(x.shape[1] == self.dim_in)
		return self.predict_model(x)

	def predict_model(self, x):
		raise Exception("'predict_model' not implemented yet!")

	def clone(self):
		raise Exception("'clone' not implemented yet!")


class AffineTransformationModel(Model):

	def __init__(self):
		Model.__init__(self, 3, 2, 2)
		self.A = None

	def fit_model(self, x, y):

		m = x.shape[0]

		Xt = np.matrix(np.zeros((2 * m, 6), dtype=np.float64))
		Y = np.matrix(y.reshape(2 * m, 1))

		for i in range(m):
			Xt[2*i,0:2] = x[i]
			Xt[2*i,2] = 1.0
			Xt[2*i+1,3:5] = x[i]
			Xt[2*i+1,5] = 1.0

		self.A = (np.linalg.inv(Xt.transpose() * Xt) * (Xt.transpose() * Y)).reshape((3,2), order = 'F')

	def predict_model(self, x):
		x_ = np.append(x, np.ones((x.shape[0], 1)), axis=1)
		return np.matrix(x_) * self.A

	def clone(self):
		cl = AffineTransformationModel()
		cl.A = None if self.A is None else self.A.copy()
		return cl

class ProjectiveTransformationModel(Model):

	def __init__(self):
		Model.__init__(self, 4, 2, 2)
		self.A = None
		self.B = None

	def fit_model(self, x, y):

		m = x.shape[0]

		Xt = np.matrix(np.zeros((2 * m, 8), dtype=np.float64))
		Y = np.matrix(y.reshape(2 * m, 1))

		for i in range(m):
			Xt[2*i,0:2] = x[i]
			Xt[2*i,2] = 1.0
			Xt[2*i,6] = - y[i,0] * x[i,0]
			Xt[2*i,7] = - y[i,0] * x[i,1]
			Xt[2*i+1,3:5] = x[i]
			Xt[2*i+1,5] = 1.0
			Xt[2*i+1,6] = - y[i,1] * x[i,0]
			Xt[2*i+1,7] = - y[i,1] * x[i,1]


		H = np.linalg.inv(Xt.transpose() * Xt) * (Xt.transpose() * Y)
		self.A = np.zeros((3,2), dtype=np.float64)
		self.B = np.zeros((3,2), dtype=np.float64)

		self.A[0,0] = H[0,0]
		self.A[1,0] = H[1,0]
		self.A[2,0] = H[2,0]
		self.A[0,1] = H[3,0]
		self.A[1,1] = H[4,0]
		self.A[2,1] = H[5,0]

		self.B[0] = H[6,0]
		self.B[1] = H[7,0]
		self.B[2] = 1

	def predict_model(self, x):
		x_ = np.append(x, np.ones((x.shape[0], 1)), axis=1)
		return (np.matrix(x_) * self.A) / (np.matrix(x_) * self.B)

	def clone(self):
		cl = ProjectiveTransformationModel()
		cl.A = None if self.A is None else self.A.copy()
		cl.B = None if self.B is None else self.B.copy()
		return cl
