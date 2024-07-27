import numpy as np

def randomize_weights (min:float = -1.0, max:float = 1.0, rows:int = 1, cols:int = 1):
    return np.random.uniform (low=min, high=max, size=(rows, cols))

def sigmoid (data) :
    return 1.0 / (1.0 + np.exp(-1 * data))

def format_data (line, normalization_value=1) -> tuple:
    values = line.strip().split(",")
    target = int(values[0])
    input = np.array(values[1:]) / normalization_value
    input = input.reshape(-1, 1)
    return (input, target)