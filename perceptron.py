import numpy as np
from util import randomize_weights

INITIAL_WEIGHT_MAX = 0.5
INITIAL_WEIGHT_MIN = -INITIAL_WEIGHT_MAX

NORMALIZATION = 255.0

LEARNING_RATE = 0.1

DEBUGGING = False


class Perceptron:
    def __init__(self, num_inputs:int , num_outputs: int, learning_rate: float =LEARNING_RATE):
        # Initialize weights using a minimum and maximum weight value.
        self.weights = randomize_weights(INITIAL_WEIGHT_MIN, INITIAL_WEIGHT_MAX, num_inputs + 1, num_outputs)
        self.learning_rate = learning_rate
        self.confusion_matrix = np.zeros ((num_outputs, num_outputs), dtype=int)
    
    def __str__ (self) -> str:
        return str(self.confusion_matrix).replace("[", " ").replace("]", " ")

    @property
    def accuracy (self) -> float:
        # Return the correct guesses divided by the total guesses.
        return ( np.trace(self.confusion_matrix) / np.sum(self.confusion_matrix) ) * 100
    
    def clear (self) -> None:
        self.confusion_matrix.fill(0)

    def activation (self, array):
        return np.where (array > 0, 1, 0)
    
    # Take in a line, parse the target value, and the input vector.
    def parse_input_line (self, line) -> tuple:
        target = int(line[0])
        input = np.array(line[1:], dtype=float) / NORMALIZATION
        return (target, input)

    def train (self, input_data, epochs:int = 1) -> None:

        for _ in range(epochs):
            
            # Loop through the input data
            for line in input_data:

                # Parse line
                target_value, input = self.parse_input_line(line)

                # Append a 1 for the bias node
                input = np.append(input, 1)

                # Create a target vector of zeroes
                target = np.zeros(self.weights.shape[1], dtype=float)

                # Set the index of the target vector that corresponds to the target_value to one.
                target[target_value] = 1

                if DEBUGGING:
                    print(f"target vector: {target}")

                # Check which "neurons" fire given the inputs
                output = self.activation(np.dot(input, self.weights))
                
                # Update the weights
                error = target - output
                update_weight = self.learning_rate * np.outer(input, error)
                self.weights += update_weight

                if DEBUGGING:
                    print(f"Output: {output}")
                    print(f"Error: {error}")
                    print(f"Updates: {update_weight}")
                
        return

    def test (self, input_data) -> None:
        # Reset the confusion matrix
        self.clear()
        
        # Loop through the input data
        for line in input_data:
            
            # Parse line
            target, input = self.parse_input_line(line)

            # Append a 1 for the bias node
            input = np.append(input, 1)

            # Calculate the output of the perceptron
            output = self.activation( np.dot (input, self.weights) )
            prediction = np.argmax ( output )

            # Populate the confusion matrix
            self.confusion_matrix[target][prediction] += 1
        return