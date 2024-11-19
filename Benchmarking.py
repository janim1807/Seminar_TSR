import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from TSInterpret.InterpretabilityModels.Saliency.TSR import TSR
from matplotlib import pyplot as plt

from Evaluation import Evaluation
from Helper import check_accuracy, train_model, create_data_loaders, load_config, load_dataset
from Masking import Masking
from Models.Transformer import Transformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Benchmarking:
    def __init__(self, training_data_path=None, training_meta_path=None, testing_data_path=None, testing_meta_path=None
                 , ml_model=None, saliency_method = None, use_tsr = True):
        self.config = load_config()
        self.ml_model = ml_model
        self.training_data_path = training_data_path
        self.training_meta_path = training_meta_path
        self.testing_data_path = testing_data_path
        self.testing_meta_path = testing_meta_path
        self.saliency_method = saliency_method
        self.use_tsr = use_tsr
        self.explain_model()

    def explain_model(self):
        config = self.config


        if self.training_data_path is None:
            self.training_data_path = "Dataset/SimulatedTrainingData_Middle_AutoRegressive_F_50_TS_50.npy"
            self.training_meta_path = "Dataset/SimulatedTrainingMetaData_Middle_AutoRegressive_F_50_TS_50.npy"
            self.testing_data_path = "Dataset/SimulatedTestingData_Middle_AutoRegressive_F_50_TS_50.npy"
            self.testing_meta_path = "Dataset/SimulatedTestingMetaData_Middle_AutoRegressive_F_50_TS_50.npy"

        num_features, num_time_steps, testing_label, testing_rnn, training_label, training_rnn, testing = load_dataset(
            self.training_data_path, self.training_meta_path, self.testing_data_path, self.testing_meta_path)

        train_loader, test_loader = create_data_loaders(training_rnn, training_label, testing_rnn, testing_label)

        num_classes = config['num_classes']
        rnndropout = config['rnndropout']
        learning_rate = config['learning_rate']
        num_epochs = config['num_epochs']
        heads = config['heads']
        n_layers = config['n_layers']

        if self.saliency_method:
            method = self.saliency_method
        else:
            method = config['method']

        if self.ml_model is None:
            model = Transformer(num_features, num_time_steps, n_layers, heads, rnndropout,
                                num_classes, time=num_time_steps).to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            train_model(model, train_loader, test_loader, criterion, num_epochs, num_time_steps, num_features,
                        optimizer)
            print("Loading model")
            model_best_name = "models/transformer_best"
            model_to_explain = torch.load(model_best_name, map_location=device)
            model.load_state_dict(model_to_explain.state_dict())
        else:
            model = self.ml_model

        start_marker = "SimulatedTrainingData"
        end_marker = "_F"

        start_index = self.training_data_path.index(start_marker) + len(start_marker)
        end_index = self.training_data_path.index(end_marker, start_index)

        data_name = self.training_data_path[start_index:end_index]
        test_acc = check_accuracy(test_loader, model, num_time_steps, num_features)
        print('{} Model: {} Test_Acc {:.4f}'.format(data_name, model.__class__.__name__, test_acc))

        if test_acc >= 90:
            method_dict = {
                'GradFlag': 'GRAD',
                'IGFlag': 'IG',
                'DLFlag': 'DL',
                'GSFlag': 'GS',
                'DLSFlag': 'DLS',
                'SGFlag': 'SG',
                'ShapleySamplingFlag': 'SVS',
                'FeatureAblationFlag': 'FA',
                'OcclusionFlag': 'FO'
            }
            tsr_method = method_dict.get(method)
            column_names = ["Saliency_Method"]
            summary_table_col = ['Datasets', 'Saliency_Methods', 'AUP', 'AUR', 'AUPR', 'AUC']

            for percentage in range(10, 100, 10):
                column_names.append(str(percentage))

            precision_results_list = pd.DataFrame(columns=column_names)
            recall_results_list = pd.DataFrame(columns=column_names)

            summary_table = pd.DataFrame(columns=summary_table_col)
            iterations = config['iterations']
            if tsr_method:
                for i in range(iterations):
                    int_mod = TSR(model, num_time_steps, num_features, method=tsr_method, mode='time')
                    item = np.array([training_rnn[i, :, :]])
                    label = int(np.argmax(training_label[i]))

                    exp = int_mod.explain(item, labels=label, TSR=self.use_tsr)
                    exp = exp[np.newaxis, :]

                    int_mod.plot(np.array([training_rnn[i, :, :]]), exp)
                    plt.show()

                    masking = Masking(exp, model, self.training_data_path, self.training_meta_path,
                                      self.testing_data_path,
                                      self.testing_meta_path, tsr_method, data_name)
                    masking.main()

                    evaluation = Evaluation(exp, model, self.training_data_path,
                                            self.training_meta_path, self.testing_data_path, self.testing_meta_path,
                                            tsr_method, data_name, i)

                    precision_results, recall_results = evaluation.precision_recall()

                    precision_results_list.loc[len(precision_results_list)] = [tsr_method] + list(precision_results)
                    recall_results_list.loc[len(recall_results_list)] = [tsr_method] + list(recall_results)

                    summary_result = evaluation.evaluate()

                    summary_table.loc[len(summary_table)] = summary_result[0]

                precision_file = f"Precision_Recall/Precision_{data_name}_{model.__class__.__name__}_{tsr_method}_TSR_{self.use_tsr}_rescaled.csv"
                recall_file = f"Precision_Recall/Recall_{data_name}_{model.__class__.__name__}_{tsr_method}_TSR_{self.use_tsr}_rescaled.csv"
                precision_results_list.to_csv(precision_file, index=False)
                recall_results_list.to_csv(recall_file, index=False)

                summary_table_file = f"Accuracy_Metrics/{data_name}_{model.__class__.__name__}_{tsr_method}_TSR_{self.use_tsr}_performance_summary.csv"

                summary_table.to_csv(summary_table_file, index=False)
                print("Total finish")
            else:
                print('Invalid method, try again:', method)
        else:
            print("Model Accuracy is not high enough")
