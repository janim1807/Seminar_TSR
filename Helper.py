import torch
import yaml
from matplotlib import pyplot as plt

from Dataset import Dataset

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch
import torch.nn as nn
import numpy as np
import timesynth as ts
import pandas as pd
import torch.utils.data as data_utils


def check_accuracy(test_loader, model, NumTimeSteps, NumFeatures, isCNN=False, returnLoss=False):
    model.eval()

    correct = 0
    total = 0
    if returnLoss:
        loss = 0
        criterion = nn.CrossEntropyLoss()

    for (samples, labels) in test_loader:
        if isCNN:
            images = samples.reshape(-1, 1, NumTimeSteps, NumFeatures).to(device, dtype=torch.float32)
        else:
            images = samples.reshape(-1, NumTimeSteps, NumFeatures).to(device, dtype=torch.float32)

        outputs = model(images.to(torch.float32))  # Convert images to torch.float32 here

        if returnLoss:
            labels = labels.to(device)
            loss += criterion(outputs, labels).data

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels.to(device)).sum()
    if returnLoss:
        loss = loss / len(test_loader)
        return (100 * float(correct) / total), loss

    return (100 * float(correct) / total)


def train_model(model_to_explain, train_loaderRNN, test_loaderRNN, criterion, num_epochs, NumTimeSteps, NumFeatures,
                optimizerTimeAtten):
    total_step = len(train_loaderRNN)
    Train_acc_flag = False
    Train_Acc = 0
    Test_Acc = 0
    BestAcc = 0
    BestEpochs = 0
    patience = 200
    saveModelBestName = "Models/transformer_best"
    saveModelLastName = "Models/transformer_last"

    for epoch in range(num_epochs):
        noImprovementflag = True
        for i, (samples, labels) in enumerate(train_loaderRNN):
            model_to_explain.train()
            samples = samples.reshape(-1, NumTimeSteps, NumFeatures).to(device)
            samples = samples.double()
            model_to_explain = model_to_explain.double()  # Convert the model's weights to double data type
            labels = labels.to(device)
            labels = labels.long()

            outputs = model_to_explain(samples)
            loss = criterion(outputs, labels)

            optimizerTimeAtten.zero_grad()
            loss.backward()
            optimizerTimeAtten.step()

            if (i + 1) % 3 == 0:
                Test_Acc = check_accuracy(test_loaderRNN, model_to_explain, NumTimeSteps, NumFeatures)
                Train_Acc = check_accuracy(train_loaderRNN, model_to_explain, NumTimeSteps, NumFeatures)
                if (Test_Acc > BestAcc):
                    BestAcc = Test_Acc
                    BestEpochs = epoch + 1
                    torch.save(model_to_explain, saveModelBestName)
                    noImprovementflag = False

                print(
                    "Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}, "
                    "Train Acc: {:.4f}, Test Acc: {:.4f}, Best Epochs: {}, "
                    "Best Acc: {:.4f}, Patience: {}".format(
                        epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                        Train_Acc, Test_Acc, BestEpochs, BestAcc, patience
                    )
                )

            if (Train_Acc >= 99 or BestAcc >= 99):
                torch.save(model_to_explain, saveModelBestName)
                Train_acc_flag = True
                break

        if (noImprovementflag):
            patience -= 1
        else:
            patience = 200

        if (epoch + 1) % 10 == 0:
            torch.save(model_to_explain, saveModelLastName)

        if (Train_acc_flag or patience == 0):
            break

        Train_Acc = check_accuracy(train_loaderRNN, model_to_explain, NumTimeSteps, NumFeatures)
        print("Best Epochs: {}, Best Accuracy: {:.4f}, Train Accuracy: {:.4f}".format(
            BestEpochs, BestAcc, Train_Acc
        ))


# Für Erstellung Masks
def get_index_of_allhighest_salient_values(array, percentageArray):
    X = array.shape[0]
    # index=np.argpartition(array, int(-1*X))
    index = np.argsort(array)
    totalSaliency = np.sum(array)
    indexes = []
    X = 1
    for percentage in percentageArray:
        actualPercentage = percentage / 100

        index_X = index[int(-1 * X):]

        percentageDroped = np.sum(array[index_X]) / totalSaliency
        if (percentageDroped < actualPercentage):
            X_ = X + 1
            index_X_ = index[int(-1 * X_):]
            percentageDroped_ = np.sum(array[index_X_]) / totalSaliency
            if (not (percentageDroped_ > actualPercentage)):
                while (percentageDroped < actualPercentage and X < array.shape[0] - 1):
                    X = X + 1
                    index_X = index[int(-1 * X):]
                    percentageDroped = np.sum(array[index_X]) / totalSaliency
        elif (percentageDroped > actualPercentage):
            X_ = X - 1
            index_X_ = index[int(-1 * X_):]
            percentageDroped_ = np.sum(array[index_X_]) / totalSaliency
            if (not (percentageDroped_ < actualPercentage)):

                while (percentageDroped > actualPercentage and X > 1):
                    X = X - 1
                    index_X = index[int(-1 * X):]
                    percentageDroped = np.sum(array[index_X]) / totalSaliency

        indexes.append(index_X)
    return indexes


def get_indices_of_highest_salient_values(array, percentage_array):
    indices = []
    total_saliency = np.sum(array)

    if total_saliency == 0:
        return indices

    sorted_indices = np.argsort(array)

    for percentage in percentage_array:
        actual_percentage = percentage / 100
        cumulative_saliency = 0
        index_list = []

        for i in range(array.shape[0] - 1, -1, -1):
            cumulative_saliency += array[sorted_indices[i]]
            index_list.append(sorted_indices[i])

            if cumulative_saliency / total_saliency >= actual_percentage:
                break

        indices.append(index_list[::-1])

    return indices


# Für Berechung MaskedAccuracy


def generate_new_sample(DataGenerationProcess, NumTimeSteps, NumFeatures, Sampler, frequency, kernel, ar_param, order,
                        hasNoise):
    if (DataGenerationProcess == None):
        sample = np.random.normal(0, 1, [NumTimeSteps, NumFeatures])

    else:
        time_sampler = ts.TimeSampler(stop_time=20)
        sample = np.zeros([NumTimeSteps, NumFeatures])

        if (Sampler == "regular"):
            time = time_sampler.sample_regular_time(num_points=NumTimeSteps * 2, keep_percentage=50)
        else:
            time = time_sampler.sample_irregular_time(num_points=NumTimeSteps * 2, keep_percentage=50)

        # different
        for i in range(NumFeatures):
            if (DataGenerationProcess == "Harmonic"):
                signal = ts.signals.Sinusoidal(frequency)

            elif (DataGenerationProcess == "GaussianProcess"):
                signal = ts.signals.GaussianProcess(kernel, nu=3. / 2)

            elif (DataGenerationProcess == "PseudoPeriodic"):
                signal = ts.signals.PseudoPeriodic(frequency, freqSD=0.01, ampSD=0.5)

            elif (DataGenerationProcess == "AutoRegressive"):
                signal = ts.signals.AutoRegressive([ar_param])

            elif (DataGenerationProcess == "CAR"):
                signal = ts.signals.CAR(ar_param, sigma=0.01)

            elif (DataGenerationProcess == "NARMA"):
                signal = ts.signals.NARMA(order)

            if hasNoise:
                noise = ts.noise.GaussianNoise(std=0.3)
                timeseries = ts.TimeSeries(signal, noise_generator=noise)
            else:
                timeseries = ts.TimeSeries(signal)

            feature, signals, errors = timeseries.sample(time)
            sample[:, i] = feature
    return sample


def mask_data(DataGenerationProcess, NumTimeSteps, NumFeatures, Sampler, frequency, kernel, ar_param, order, hasNoise,
              data, mask, noise=False):
    newData = np.zeros((data.shape))
    if (noise):
        noiseSample = generate_new_sample(DataGenerationProcess, NumTimeSteps, NumFeatures, Sampler, frequency, kernel,
                                          ar_param, order, hasNoise)
        noiseSample = noiseSample.reshape(data.shape[1])
    for i in range(mask.shape[0]):
        newData[i, :] = data[i, :]
        cleanIndex = mask[i, :]
        cleanIndex = cleanIndex[np.logical_not(pd.isna(cleanIndex))]
        cleanIndex = cleanIndex.astype(np.int64)
        if (noise):
            newData[i, cleanIndex] = noiseSample[cleanIndex]
        else:
            newData[i, cleanIndex] = 0

    return newData


def save_into_csv(data, file, Flip=False, col=None, index=False):
    if (Flip):
        print("Will Flip before Saving")
        data = data.reshape((data.shape[1], data.shape[0]))

    df = pd.DataFrame(data)
    if (col != None):
        df.columns = col
    df.to_csv(file, index=index)


def load_dataset(training_data_path, training_meta_path, testing_data_path, testing_meta_path):
    dataset = Dataset(training_data_path, training_meta_path, testing_data_path, testing_meta_path)
    num_features, num_time_steps, testing_label, testing_rnn, training_label, training_rnn, testing = dataset.load_Data(
        training_data_path, training_meta_path, testing_data_path, testing_meta_path)
    return num_features, num_time_steps, testing_label, testing_rnn, training_label, training_rnn, testing


def load_csv(file, returnDF=False, Flip=False):
    df = pd.read_csv(file)
    data = df.values
    if (Flip):
        print("Will Un-Flip before Loading")
        data = data.reshape((data.shape[1], data.shape[0]))
    if (returnDF):
        return df
    return data


def get_scaler(training_data_path, training_meta_path, testing_data_path, testing_meta_path):
    dataset = Dataset(training_data_path, training_meta_path, testing_data_path, testing_meta_path)
    return dataset.get_scaler(training_data_path)


def get_testing_meta_data(training_data_path, training_meta_path, testing_data_path, testing_meta_path):
    dataset = Dataset(training_data_path, training_meta_path, testing_data_path, testing_meta_path)
    return dataset.get_testing_and_testing_meta_data(testing_data_path, testing_meta_path)


def create_data_loaders(training_rnn, training_label, testing_rnn, testing_label):
    config = load_config()
    batch_size = config['batch_size']
    train_data = data_utils.TensorDataset(torch.from_numpy(training_rnn), torch.from_numpy(training_label))
    train_loader = data_utils.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data = data_utils.TensorDataset(torch.from_numpy(testing_rnn), torch.from_numpy(testing_label))
    test_loader = data_utils.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def load_config():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config


def plot_example_box(input, saveLocation, show=False, greyScale=False, flip=False):
    if (flip):
        input = np.transpose(input)
    fig, ax = plt.subplots()

    if (greyScale):
        cmap = 'gray'
    else:
        cmap = 'seismic'

    plt.axis('off')

    cax = ax.imshow(input, interpolation='nearest', cmap=cmap)

    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(saveLocation + '.png', bbox_inches='tight', pad_inches=0)

    if (show):
        plt.show()
    plt.close()


def get_index_of_xhighest_features(array, X):
    return np.argpartition(array, int(-1 * X))[int(-1 * X):]
