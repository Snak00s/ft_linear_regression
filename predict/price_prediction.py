import numpy as np
import sys

def estimatePrice(mileage: float, theta0=0.0, theta1=0.0) -> float:
    """Estimate the price of a car based on car mileage"""
    return theta0 + (theta1 * mileage)

def main():
    try:
        mileage = input('Enter the mileage of your car: ')
        mileage = float(mileage)
    except KeyboardInterrupt:
        exit(0)
    except ValueError:
        print('Error: Invalid mileage')
        exit(1)
    except MemoryError:
        print("Memory error")
        exit(1)

    try:
        result_f = open('../result.txt')
        content = result_f.readlines()
        assert len(content) == 2
        assert content[0].count('theta0 = ') == 1
        assert content[1].count('theta1 = ') == 1

        str_t0 = content[0].removeprefix('theta0 = ').removesuffix('\n')
        str_t1 = content[1].removeprefix('theta1 = ').removesuffix('\n')
        t0 = float(str_t0)
        t1 = float(str_t1)

        print('Estimated price of your car:',
                estimatePrice(mileage, theta0=t0, theta1=t1))
    except ValueError:
        print('Error: Invalid theta value')
        exit(1)
    except FileNotFoundError:
        print('No \'result.txt\' found')
        print('Estimated price of your car:',
                estimatePrice(mileage, theta0=t0, theta1=t1))
        exit(0)
    except AssertionError:
        print("Invalid file format")
    except MemoryError:
        print("Memory error")
        exit(1)
    return

if __name__ == "__main__":
    main()