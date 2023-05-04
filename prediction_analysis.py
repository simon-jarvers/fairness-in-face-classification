import numpy as np

if __name__ == "__main__":
    filename = "predictions_config_test_2023-05-04_10-29-34_616881.pkl"
    prediction = np.load(filename, allow_pickle=True)
    print("Prediction:")
    print(len(prediction))
    print(prediction[0].size())
    print(prediction[-2].size())
    print(prediction[-1].size())
    filename = "groundtruth_config_test_2023-05-04_10-29-34_616881.pkl"
    groundtruth = np.load(filename, allow_pickle=True)
    print("Ground truth")
    print(len(groundtruth))
    print(groundtruth[0].size())
    print(groundtruth[-2].size())
    print(groundtruth[-1].size())

