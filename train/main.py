import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def norm_robust(data: list, median :float, iqr: float):
	"""Normalize value using robust scaler"""
	return ([(x - median) / iqr for x in data])

def norm_minmax(data: list) -> list:
	"""Normalize value using minmaxing"""
	data_min = min(data)
	data_max = max(data)
	return ([(x - data_min) / (data_max - data_min) for x in data])

def estimatePrice(mileage: float, theta0=0.0, theta1=0.0) -> float:
    """Estimate the price of a car based on car mileage"""
    return theta0 + (theta1 * mileage)

def t0_sums(mileage: list, price: list, t0=0.0, t1=0.0) -> float:
	"""Do the sums of the theta0 formula"""
	return sum([(estimatePrice(mileage[i], t0, t1) - price[i]) for i in range(len(mileage))])

def t1_sums(mileage: list, price: list, t0=0.0, t1=0.0) -> float:
	"""Do the sums of the theta1 formula"""
	return sum([((estimatePrice(mileage[i], t0, t1) - price[i]) * mileage[i]) for i in range(len(mileage))])

def calculate_theta(mileage: list, price: list, t0=0.0, t1=0.0) -> list:
	"""Calculate sums of theta0 and theta1 formula, return a list with the result : [result0, result1]"""
	return ([t0_sums(mileage, price, t0, t1), t1_sums(mileage, price, t0, t1)])

def lin_reg_minmax(mileage: list, price: list, max_iter: int, learning_rate: float) -> list:
	"""Linear regression using gradient descent, optimize it with minmax scaler"""
	try:
		assert type(mileage) == list
		assert type(price) == list
		assert type(max_iter) == int
		assert type(learning_rate) == float or type(learning_rate) == int
	except AssertionError:
		print('Error: Invalid argument type: (list, list, int, float/int) expected')
		return None
	min_mil = min(mileage)
	max_mil = max(mileage)
	n_mileage = norm_minmax(mileage)

	theta0 = 0.0
	theta1 = 0.0
	data_len = float(len(mileage))
	for _ in range(max_iter):
		theta_list = calculate_theta(n_mileage, price, theta0, theta1)
		theta_list = [(1 / data_len) * theta for theta in theta_list]

		theta0 = theta0 - (learning_rate * theta_list[0])
		theta1 = theta1 - (learning_rate * theta_list[1])

	theta1 = theta1 / (max_mil - min_mil)
	theta0 = theta0 - theta1 * min_mil
	plt.plot([min_mil, max_mil], [estimatePrice(min_mil, theta0, theta1), estimatePrice(max_mil, theta0, theta1)], 'r-', label='Minmax')
	return([theta0, theta1])

def lin_reg_robust(mileage: list, price: list, max_iter: int, learning_rate: float) -> list:
	"""Linear regression using gradient descent, optimize it with robust scaler"""
	try:
		assert type(mileage) == list
		assert type(price) == list
		assert type(max_iter) == int
		assert type(learning_rate) == float or type(learning_rate) == int
	except AssertionError:
		print('Error: Invalid argument type: (list, list, int, float/int) expected')
		return None
	median = np.median(mileage)
	iqr = np.quantile(mileage, 0.75) - np.quantile(mileage, 0.25)
	n_mileage = norm_robust(mileage, median, iqr)

	theta0 = 0.0
	theta1 = 0.0
	data_len = float(len(mileage))
	for _ in range(max_iter):
		theta_list = calculate_theta(n_mileage, price, theta0, theta1)
		theta_list = [(1 / data_len) * theta for theta in theta_list]

		theta0 = theta0 - (learning_rate * theta_list[0])
		theta1 = theta1 - (learning_rate * theta_list[1])

	theta1 = theta1 / iqr
	theta0 = theta0 - theta1 * median
	min_mil = min(mileage)
	max_mil = max(mileage)
	plt.plot([min_mil, max_mil], [estimatePrice(min_mil, theta0, theta1), estimatePrice(max_mil, theta0, theta1)], 'b-', label='Robust')
	return([theta0, theta1])

def main():
	try:
		data = pd.read_csv('../data.csv')
		mileage = list(data['km'])
		price = list(data['price'])
		plt.scatter(x=mileage, y=price)

		max_iter = 500
		learning_rate = 0.1
		# result_r = lin_reg_robust(mileage, price, max_iter, learning_rate)
		# assert type(result_r) == list
		result_mm = lin_reg_minmax(mileage, price, max_iter, learning_rate)
		assert type(result_mm) == list
		test = open('../result.txt', "w")
		test.write(f"theta0 = {result_mm[0]}\ntheta1 = {result_mm[1]}\n")
		test.close()

		plt.xlabel('mileage')
		plt.ylabel('price')
		plt.legend()
		plt.savefig('../figure.jpg')

	except FileNotFoundError:
		print('Error: No \'data.csv\' file founded')
		exit(1)
	except KeyError:
		print('Error: \'data.csv\' do not containt expected columns')
		exit(1)
	except KeyboardInterrupt:
		pass
	except PermissionError:
		print("Error: file action not allowed")
		exit(1)
	except AssertionError:
		print("Error: linear regression failed")
		exit(1)
	return

if __name__ == "__main__":
	main()