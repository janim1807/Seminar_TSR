import torch

from Benchmarking import Benchmarking

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

training_data_path = "Dataset/SimulatedTrainingData_Middle_AutoRegressive_F_50_TS_50.npy"
training_meta_path = "Dataset/SimulatedTrainingMetaData_Middle_AutoRegressive_F_50_TS_50.npy"
testing_data_path = "Dataset/SimulatedTestingData_Middle_AutoRegressive_F_50_TS_50.npy"
testing_meta_path = "Dataset/SimulatedTestingMetaData_Middle_AutoRegressive_F_50_TS_50.npy"


#training_data_path = "Dataset/SimulatedTrainingData_Middle_GaussianProcess_F_50_TS_50.npy"
#training_meta_path = "Dataset/SimulatedTrainingMetaData_Middle_GaussianProcess_F_50_TS_50.npy"
#testing_data_path = "Dataset/SimulatedTestingData_Middle_GaussianProcess_F_50_TS_50.npy"
#testing_meta_path = "Dataset/SimulatedTestingMetaData_Middle_GaussianProcess_F_50_TS_50.npy"

save_model_Transformer = "Models/transformer_best"


model_to_explain = torch.load(save_model_Transformer, map_location=device)

Benchmarking(training_data_path, training_meta_path, testing_data_path, testing_meta_path)
