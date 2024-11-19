import numpy as np
from sklearn.preprocessing import MinMaxScaler


class Dataset:
    def __init__(self, training_data_path, training_meta_path, testing_data_path, testing_meta_path):
        self.NumFeatures, self.NumTimeSteps, self.TestingLabel, self.TestingRNN, self.TrainingLabel, self.TrainingRNN, self.Testing = self.load_Data(
            training_data_path, training_meta_path, testing_data_path, testing_meta_path)

    def load_Data(self, training_data_path, training_meta_path, testing_data_path, testing_meta_path):
        Training = np.load(training_data_path)
        TrainingMetaDataset = np.load(training_meta_path)
        TrainingLabel = TrainingMetaDataset[:, 0]
        Testing = np.load(testing_data_path)
        TestingDataset_MetaData = np.load(testing_meta_path)
        TestingLabel = TestingDataset_MetaData[:, 0]

        Training = Training.reshape(Training.shape[0], Training.shape[1] * Training.shape[2])
        Testing = Testing.reshape(Testing.shape[0], Testing.shape[1] * Testing.shape[2])

        scaler = MinMaxScaler()
        scaler.fit(Training)
        Training = scaler.transform(Training)
        Testing = scaler.transform(Testing)

        NumTimeSteps = 50
        NumFeatures = 50

        TrainingRNN = Training.reshape(Training.shape[0], NumTimeSteps, NumFeatures)
        TestingRNN = Testing.reshape(Testing.shape[0], NumTimeSteps, NumFeatures)

        return NumFeatures, NumTimeSteps, TestingLabel, TestingRNN, TrainingLabel, TrainingRNN, Testing

    def get_scaler(self, training_data_path):
        Training = np.load(training_data_path)
        Training = Training.reshape(Training.shape[0], Training.shape[1] * Training.shape[2])
        scaler = MinMaxScaler()
        scaler = scaler.fit(Training)

        return scaler

    def get_testing_and_testing_meta_data(self, testing_data_path, testing_meta_path):
        testing = np.load(testing_data_path)
        TestingDataset_MetaData = np.load(testing_meta_path)
        return testing, TestingDataset_MetaData
