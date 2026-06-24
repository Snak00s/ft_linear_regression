import pandas as pd
import matplotlib.pyplot as plt

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

def lin_reg(mileage: list, price: list) -> list:
	"""Linear regression using gradient descent, optimize it with minmax normalization"""
	min_mil = min(mileage)
	max_mil = max(mileage)
	n_mileage = norm_minmax(mileage)

	theta0 = 0.0
	theta1 = 0.0
	max_iter = 1000
	learning_rate = 0.1
	data_len = float(len(mileage))
	for iter in range(max_iter):
		theta_list = calculate_theta(n_mileage, price, theta0, theta1)
		theta_list = [(1 / data_len) * theta for theta in theta_list]

		theta0 = theta0 - (learning_rate * theta_list[0])
		theta1 = theta1 - (learning_rate * theta_list[1])

	theta1 = theta1 / (max_mil - min_mil)
	theta0 = theta0 - theta1 * min_mil
	plt.plot([min_mil, max_mil], [estimatePrice(min_mil, theta0, theta1), estimatePrice(max_mil, theta0, theta1)])
	return([theta0, theta1])

def main():
	try:
		data = pd.read_csv('../data.csv')
		mileage = list(data['km'])
		price = list(data['price'])
		plt.scatter(x=mileage, y=price)
		result = lin_reg(mileage, price)
		test = open('../result.txt', "w")
		test.write(f"theta0 = {result[0]}\ntheta1 = {result[1]}")
		test.close()
		plt.show()
	except FileNotFoundError:
		print('Error: No \'data.csv\' file founded')
		exit(1)
	except KeyError:
		print('Error: csv file do not containt correct columns')
		exit(1)
	except KeyboardInterrupt:
		pass
	return

if __name__ == "__main__":
	main()