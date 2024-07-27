import numpy as np
from perceptron import Perceptron

from gradient_descent import gradient_descent

NORMALIZE_VALUE = 255.0
TRAINING_ITERATIONS = 1

DEFAULT_NETWORK_INPUTS = 784
DEFAULT_NETWORK_OUTPUTS = 10

LEARNING_RATE_1 = 0.1
LEARNING_RATE_2 = 0.01
LEARNING_RATE_3 = 0.001

def main ():
    print("Uploading Files")
    training_file = np.genfromtxt("mnist_train.csv", delimiter=',', skip_header=1)
    testing_file = np.genfromtxt("mnist_test.csv", delimiter=',', skip_header=1)
    print("Finished Uploading Files")

    P = Perceptron(DEFAULT_NETWORK_INPUTS, DEFAULT_NETWORK_OUTPUTS, LEARNING_RATE_1)

    epoch_count = 0

    while epoch_count < 70:
        P.train(training_file)
        P.test(testing_file)
        print(f"Epoch {epoch_count} Accuracy: {P.accuracy}")
        epoch_count += 1
    
    print(f"\n\nAchieved {P.accuracy} in {epoch_count} epochs.")
    print(P)

if __name__ == "__main__":
    main()