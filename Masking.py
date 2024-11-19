import re
import time

import numpy as np
import torch
import torch.utils.data as data_utils

from Helper import get_index_of_allhighest_salient_values, create_data_loaders, check_accuracy, load_config, mask_data, \
    generate_new_sample, plot_example_box, save_into_csv, get_scaler, load_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

percentages = [i for i in range(0, 101, 10)]
maskedPercentages = [i for i in range(0, 101, 10)]
DataGenerationTypes = [None, "Harmonic", "GaussianProcess", "PseudoPeriodic", "AutoRegressive", "CAR", "NARMA"]


class Masking:
    def __init__(self, exp, model, training_data_path, training_meta_path, testing_data_path, testing_meta_path,
                 tsr_method, data_name):
        self.saliency_methods = [tsr_method]
        self.exp = exp
        self.model = model
        self.training_data_path = training_data_path
        self.training_meta_path = training_meta_path
        self.testing_data_path = testing_data_path
        self.testing_meta_path = testing_meta_path
        self.data_name = data_name

    def main(self):
        self.create_masks()
        self.get_masked_accuracy()

    def create_masks(self):
        model_name = self.model.__class__.__name__
        for saliency in self.saliency_methods:
            saliency_values = self.exp.reshape(self.exp.shape[0], -1)
            index_grid = np.zeros((saliency_values.shape[0], saliency_values.shape[1], len(percentages)),
                                  dtype='object')
            index_grid[:, :, :] = np.nan
            for i in range(saliency_values.shape[0]):
                indexes = get_index_of_allhighest_salient_values(saliency_values[i, :], percentages)
                for l in range(len(indexes)):
                    index_grid[i, :len(indexes[l]), l] = indexes[l]
            for p, percentage in enumerate(percentages):
                np.save(f"MaskingData/{model_name}_{saliency}_{percentage}_percentSal_rescaled.npy",
                        index_grid[:, :, p])
        print("Creating Masks for", model_name)

    def get_masked_accuracy(self):
        num_features, num_time_steps, testing_label, testing_rnn, training_label, training_rnn, testing = load_dataset(
            self.training_data_path, self.training_meta_path, self.testing_data_path, self.testing_meta_path)
        train_loader, test_loader = create_data_loaders(training_rnn, training_label, testing_rnn, testing_label)

        np_load_old = np.load
        np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

        model_name = self.model.__class__.__name__

        start = time.time()
        y_dim_of_grid = len(maskedPercentages) + 1
        x_dim_of_grid = len(self.saliency_methods)
        grid = np.zeros((x_dim_of_grid, y_dim_of_grid), dtype='object')
        grid[:, 0] = self.saliency_methods
        columns = ["saliency method"]
        for mask in maskedPercentages:
            columns.append(str(mask))

        raw_testing = np.copy(testing)

        model = self.model

        test_unmasked_acc = check_accuracy(test_loader, model, num_time_steps, num_features)

        for s, saliency in enumerate(self.saliency_methods):
            test_masked_acc = test_unmasked_acc
            for i, masked_percentage in enumerate(maskedPercentages):
                data_generation_type = re.split("_", self.training_data_path, 1)[1]

                data_generation_process = None
                for generation_type in DataGenerationTypes:
                    if generation_type is not None and generation_type in data_generation_type:
                        data_generation_process = generation_type
                        break

                config = load_config()

                sampler = config['sampler']
                frequency = config['frequency']
                kernel = config['kernel']
                ar_param = config['ar_param']
                order = config['order']
                has_noise = config['has_noise']

                scaler = get_scaler(self.training_data_path, self.training_meta_path, self.testing_data_path,
                                    self.testing_meta_path)

                if masked_percentage == 0:
                    grid[s][i + 1] = test_unmasked_acc
                elif test_masked_acc == 0:
                    grid[s][i + 1] = test_masked_acc
                else:
                    if masked_percentage != 100:
                        mask = np.load(
                            f"MaskingData/{model_name}_{saliency}_{masked_percentage}_percentSal_rescaled.npy")
                        to_mask = np.copy(raw_testing)
                        masked_testing = mask_data(data_generation_process, num_time_steps, num_features, sampler,
                                                   frequency, kernel, ar_param, order, has_noise, to_mask,
                                                   mask, True)
                        masked_testing = scaler.transform(masked_testing)
                        masked_testing = masked_testing.reshape(-1, num_time_steps, num_features)
                    else:
                        masked_testing = np.zeros((testing.shape[0], num_time_steps * num_features))
                        sample = generate_new_sample(data_generation_process, num_time_steps, num_features,
                                                     sampler,
                                                     frequency, kernel, ar_param, order, has_noise).reshape(
                            num_time_steps * num_features)
                        masked_testing[:, :] = sample
                        masked_testing = scaler.transform(masked_testing)
                        masked_testing = masked_testing.reshape(-1, num_time_steps, num_features)

                    if True:
                        random_index = 10
                        plot_example_box(masked_testing[random_index],
                                       f"MaskingImages/{saliency}_percentMasked{masked_percentage}", flip=True)
                    masked_test_data_rnn = data_utils.TensorDataset(torch.from_numpy(masked_testing),
                                                                    torch.from_numpy(testing_label))
                    masked_test_loader_rnn = data_utils.DataLoader(masked_test_data_rnn,
                                                                   batch_size=config['batch_size'],
                                                                   shuffle=False)

                    test_masked_acc = check_accuracy(masked_test_loader_rnn, model, num_time_steps, num_features)
                    print(
                        f'{model_name} {saliency} Acc {test_unmasked_acc:.2f} Masked Acc {test_masked_acc:.2f} Highest Value mask {masked_percentage}')
                    grid[s][i + 1] = test_masked_acc
                end_percentage = time.time()
        end = time.time()
        print(f'{model_name} time: {end - start}')
        print()

        result_file_name = '--Masked_Acc_dir' + self.data_name + "_" + model_name + "_0_10_20_30_40_50_60_70_80_90_100_percentSal_rescaled.csv"
        save_into_csv(grid, result_file_name, col=columns)

        np.load = np_load_old
