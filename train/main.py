import pandas as pd
import numpy as np

def estimatePrice(mileage: float, theta0=0.0, theta1=0.0) -> float:
    """Estimate the price of a car based on car mileage"""
    return theta0 + (theta1 * mileage)

def t0_sums(mileage: list, price: list, t0=0.0, t1=0.0):
	return np.sum([estimatePrice(mileage[i], t0, t1) - price[i] for i in range(len(mileage))])

def t1_sums(mileage: list, price: list, t0=0.0, t1=0.0):
	return np.sum([(estimatePrice(mileage[i], t0, t1) - price[i]) * mileage[i] for i in range(len(mileage))])

def calculate_theta(mileage: list, price: list, t0=0.0, t1=0.0) -> list:
	return ([t0_sums(mileage, price, t0, t1), t1_sums(mileage, price, t0, t1)])

def algo(mileage: list, price: list) -> list:
	theta0 = 0
	theta1 = 0
	max_iter = 5
	learning_rate = 1
	data_len = len(mileage)
	for iter in range(max_iter):
		theta_list = calculate_theta(mileage, price, theta0, theta1)
		theta0 -= (learning_rate * (1 / data_len) * theta_list[0])
		theta1 -= (learning_rate * (1 / data_len) * theta_list[1])
	return([theta0, theta1])

def main():
	try:
		data = pd.read_csv('../data.csv')
		mileage = list(data['km'])
		price = list(data['price'])
	except FileNotFoundError:
		exit(0)
	except KeyError:
		print('Error: csv file do not containt correct columns')
		exit(0)
	print(algo(mileage, price))
	return

if __name__ == "__main__":
	main()