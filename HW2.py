import pandas as pd
import numpy as np

np.random.seed(0)

def resample(data, label, outlier_ratio=0.01, target_label=0):
    """
    Resample the data to balance classes.

    Parameters:
        data: np.array, shape=(n_samples, n_features)
            Input data.
        label: np.array, shape=(n_samples,)
            Labels corresponding to the data samples.
        outlier_ratio: float, optional (default=0.01)
            Ratio of outliers to include in the resampled data.
        target_label: int, optional (default=0)
            The label to be treated as normal.

    Returns:
        new_data: np.array
            Resampled data.
        new_label: np.array
            Resampled labels.
    """
    new_data = []
    new_label = []
    for i in [1, -1]:
        if i != target_label:
            i_data = data[label == i]
            target_size = len(data[label == target_label])
            num = target_size * outlier_ratio
            idx = np.random.choice(
                list(range(len(i_data))), int(num), replace=False
            )
            new_data.append(i_data[idx])
            new_label.append(np.ones(len(idx)) * 1)
        else:
            new_data.append(data[label == i])
            new_label.append(np.ones(len(data[label == i])) * 0)
    new_data = np.concatenate(new_data)
    new_label = np.concatenate(new_label)
    return new_data, new_label


from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# Function to visualize data using line charts
def visualize_samples(normal_samples, abnormal_samples, title):
    plt.figure(figsize=(10, 6))
    for i, sample in enumerate(normal_samples):
        plt.plot(sample, label=f'Normal Sample {i+1}')
    for i, sample in enumerate(abnormal_samples):
        plt.plot(sample, label=f'Abnormal Sample {i+1}')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

if __name__=='__main__':
    # Load the data
    category = "ECG200" # Wafer / ECG200
    print(f"Dataset: {category}")
    train_data = pd.read_csv(f'./{category}/{category}_TRAIN.tsv', sep='\t', header=None).to_numpy()
    test_data = pd.read_csv(f'./{category}/{category}_TEST.tsv', sep='\t', header=None).to_numpy()

    train_label = train_data[:, 0].flatten()
    train_data = train_data[:, 1:]
    train_data, train_label = resample(train_data, train_label, outlier_ratio=0.0, target_label=1)

    test_label = test_data[:, 0].flatten()
    test_data = test_data[:, 1:]
    test_data, test_label = resample(test_data, test_label, outlier_ratio=0.1, target_label=1)

    # Visualize random samples from training and testing data
    train_normal_samples = train_data[train_label == 0][:10]
    train_abnormal_samples = train_data[train_label == 1][:10]
    test_normal_samples = test_data[test_label == 0][:10]
    test_abnormal_samples = test_data[test_label == 1][:10]
        
    visualize_samples(train_normal_samples, f'{category} Training Normal Samples')
    visualize_samples(train_abnormal_samples, f'{category} Training Abnormal Samples')
    visualize_samples(test_normal_samples, f'{category} Testing Normal Samples')
    visualize_samples(test_abnormal_samples, f'{category} Testing Abnormal Samples')
