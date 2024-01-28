import sys
import pandas as pd
import numpy as np

def topsis(input_file, weights, impacts, output_file):
    # Read data from CSV file
    data = pd.read_csv(input_file)
    
    # Extract the decision matrix
    X = data.iloc[:, 1:].values
    
    # Normalize the decision matrix
    normalized_matrix = X / np.sqrt((X**2).sum(axis=0))
    
    # Multiply normalized matrix by the weight matrix
    weighted_matrix = normalized_matrix * weights
    
    # Determine the positive and negative ideal solutions
    ideal_positive = weighted_matrix.max(axis=0)
    ideal_negative = weighted_matrix.min(axis=0)
    
    # Calculate the separation measures
    separation_positive = np.sqrt(((weighted_matrix - ideal_positive)**2).sum(axis=1))
    separation_negative = np.sqrt(((weighted_matrix - ideal_negative)**2).sum(axis=1))
    
    # Calculate the performance score
    performance_score = separation_negative / (separation_positive + separation_negative)
    
    # Determine the rank
    rank = len(performance_score) - np.argsort(performance_score) 
    
    # Write the result to the output file
    result_data = data.iloc[:, :].reset_index(drop=True)
    result_data['Score'] = performance_score
    result_data['Rank'] = rank
    result_data.to_csv(output_file, index=False)

if __name__ == "__main__":
    # Check if the correct number of command linex arguments is provided
    if len(sys.argv) != 5:
        print("Usage: python filename input.csv <weights> <impacts> result.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    weights = np.array(list(map(float, sys.argv[2].split(','))))
    impacts = np.array([i for i in sys.argv[3].split(',')])
    output_file = sys.argv[4]

    # Validate the input lengths
    if len(weights) != len(impacts):
        print("Error: Number of weights and impacts must be the same.")
        sys.exit(1)
    
    # Run TOPSIS
    topsis(input_file, weights, impacts, output_file)
    
    print(f"TOPSIS completed successfully. Results saved to {output_file}")
