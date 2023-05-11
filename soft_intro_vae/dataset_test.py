import numpy as np

def print_dataset(file_path, n=10):  # Print the first n items
    data = np.load(file_path)
    print("Size of the entire dataset:", data.shape)
    print(f"Contents of the first {n} items in the dataset:")
    for i in range(min(len(data), n)):
        print(f"Item {i}:")
        print(data[i])

# Replace 'your_file_path.npy' with the actual path to your file
print_dataset('D:\\GitHub\\T2I-Adapter\\preprocessed_outputs\\val_stim_multi_trial_data-color.npy', 10)