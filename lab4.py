import pandas as pd

def find_s(file_path):
    data = pd.read_csv(file_path)
    print("Training data:\n", data)
    
    first_positive = data[data.iloc[:, -1] == 'Yes'].iloc[0]
    h = list(first_positive.iloc[:-1]) 
    
    for _, r in data[data.iloc[:, -1] == 'Yes'].iloc[1:].iterrows():
        h = [hv if hv == rv else '?' for hv, rv in zip(h, r.iloc[:-1])]
    
    return h

path = r"training_data.csv"
final_h = find_s(path)
print("\nFinal hypothesis:", final_h)
