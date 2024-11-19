# Compute Precision and Recall
import time

import numpy as np
import pandas as pd

from Helper import get_testing_meta_data, get_index_of_xhighest_features, load_csv, save_into_csv, load_config


class Evaluation:

    def main(self):
        self.precision_recall()
        self.evaluate()

    def __init__(self, exp,model, training_data_path, training_meta_path, testing_data_path,
                 testing_meta_path, tsr_method, data_name, i):
        self.maskedPercentages = [i for i in range(0, 101, 10)]
        self.model = model
        self.exp = exp
        self.saliency_methods = [tsr_method]
        self.training_data_path = training_data_path
        self.training_meta_path = training_meta_path
        self.testing_data_path = testing_data_path
        self.testing_meta_path = testing_meta_path
        self.model_name = self.model.__class__.__name__
        self.data_name = data_name
        self.i = i

    def precision_recall(self):
        column_names = ["Saliency_Methods"]
        for percentage in range(0, 100, 10):
            column_names.append(str(percentage))

        testing, testing_dataset_metadata = get_testing_meta_data(self.training_data_path,
                                                                         self.training_meta_path,
                                                                         self.testing_data_path,
                                                                         self.testing_meta_path)
        # TODO Improve datasets calls
        num_features = 50
        num_time_steps = 50

        testing_labels = testing_dataset_metadata[:, 0]
        target_ts_starts = testing_dataset_metadata[:, 1]
        target_ts_ends = testing_dataset_metadata[:, 2]
        target_feat_starts = testing_dataset_metadata[:, 3]
        target_feat_ends = testing_dataset_metadata[:, 4]

        references_samples = np.zeros(testing.shape)
        reference_idx_all = np.zeros((testing.shape[0], num_time_steps * num_features))
        reference_idx_all[:, :] = np.nan

        for i in range(testing_labels.shape[0]):
            references_samples[i, int(target_ts_starts[i]):int(target_ts_ends[i]),
            int(target_feat_starts[i]):int(target_feat_ends[i])] = 1
            number_of_imp_features = int(np.sum(references_samples[i, :, :]))
            ind = get_index_of_xhighest_features(references_samples[i, :, :].flatten(), number_of_imp_features)
            reference_idx_all[i, :ind.shape[0]] = ind.reshape(-1)

        np_load_old = np.load
        np.load = lambda *a, **k: np_load_old(*a, **k)

        box_start_time = time.time()

        precision_results = pd.DataFrame(columns=column_names)
        recall_results = pd.DataFrame(columns=column_names)
        precision_results["Saliency_Methods"] = self.saliency_methods
        recall_results["Saliency_Methods"] = self.saliency_methods

        start = time.time()
        for s, saliency_method in enumerate(self.saliency_methods):
            precision = []
            recall = []
            saliency_values = self.exp
            saliency_values = saliency_values.reshape(-1, num_features * num_time_steps)

            for percentage in range(0, 100, 10):
                overall_recall = 0
                overall_precision = 0

                if percentage != 100 and percentage != 0:
                    filename = f"MaskingData/{self.model_name}_{saliency_method}_{percentage}_percentSal_rescaled.npy"
                    mask = np.load(filename, allow_pickle=True)
                    mask = mask.reshape(-1, mask.shape[0] * mask.shape[1])

                    recall_count = 0
                    precision_count = 0

                    for i in range(mask.shape[0]):
                        positive_index = mask[i, :]
                        positive_index = positive_index[np.logical_not(pd.isna(positive_index))]
                        positive_index = positive_index.astype(np.int64)
                        true_index = reference_idx_all[i][:]
                        true_index = true_index[np.logical_not(np.isnan(true_index))]
                        true_index = true_index.astype(np.int64)

                        positive_with_true = np.isin(positive_index, true_index)
                        true_with_positive = np.isin(true_index, positive_index)

                        count_tp = 0
                        count_fp = 0
                        count_fn = 0

                        tp = 0
                        fp = 0
                        fn = 0

                        for j in range(positive_with_true.shape[0]):
                            if positive_with_true[j]:
                                # In positive and true, so true positive
                                tp += saliency_values[i, positive_index[j]]
                                count_tp += 1
                            else:
                                # In positive but not true, so false positive
                                fp += saliency_values[i, positive_index[j]]
                                count_fp += 1
                        for j in range(true_with_positive.shape[0]):
                            if not true_with_positive[j]:
                                # In true but not in positive, false negative
                                fn += saliency_values[i, true_index[j]]
                                count_fn += 1

                        if tp + fp > 0:
                            example_precision = tp / (tp + fp)
                            precision_count += 1
                        else:
                            example_precision = 0
                        if tp + fn > 0:
                            example_recall = tp / (tp + fn)
                            recall_count += 1
                        else:
                            example_recall = 0

                        overall_precision += example_precision
                        overall_recall += example_recall

                    overall_precision = overall_precision / precision_count
                    overall_recall = overall_recall / recall_count
                    precision.append(overall_precision)
                    recall.append(overall_recall)
                else:
                    precision.append(np.nan)
                    recall.append(np.nan)

                print(f"{self.data_name} {self.model_name} {saliency_method} masked percentages {percentage} "
                      f"Precision {overall_precision:.4f} Recall {overall_recall:.4f}")

            precision_results.loc[s, 1:] = precision
            recall_results.loc[s, 1:] = recall

        end = time.time()
        print(f"{self.data_name}_{self.model_name}", end - start)

        precision_file = f"Precision_Recall/Precision_{self.data_name}_{self.model_name}_rescaled.csv"
        recall_file = f"Precision_Recall/Recall_{self.data_name}_{self.model_name}_rescaled.csv"
        precision_results.to_csv(precision_file, index=False)
        recall_results.to_csv(recall_file, index=False)

        box_end_time = time.time()
        print("Finish", self.data_name, box_end_time - box_start_time)

        np.load = np_load_old

        return precision_results.loc[0, '10':'90'].values.astype(float),  recall_results.loc[0, '10':'90'].values.astype(float)


    # Compute Accuracy Metrics AUP, AUR, AUPR

    def evaluate(self):
        precision_table_col = ['AUP', 'models', 'Saliency_Methods', 'Datasets']
        recall_table_col = ['AUR', 'models', 'Saliency_Methods', 'Datasets']
        aupr_table_col = ['AUPR', 'models', 'Saliency_Methods', 'Datasets']
        auc_table_col = ['AUC', 'models', 'Saliency_Methods', 'Datasets']
        summary_table_col = ['Datasets', 'Saliency_Methods', 'AUP', 'AUR', 'AUPR', 'AUC']

        config = load_config()
        Timeflag = config['timeflag']
        Featureflag = config['featureflag']

        if True:
            precision_table = []
            recall_table = []
            aupr_table = []

            time_precision_recall = Timeflag
            feature_precision_recall = Featureflag

            if time_precision_recall:
                time_precision_table = []
                time_recall_table = []
                time_aupr_table = []

            if feature_precision_recall:
                feature_precision_table = []
                feature_recall_table = []
                feature_aupr_table = []

            auc_table = []
            summary_table = []

            precision_file = 'Precision_Recall/' + "Precision_" + self.data_name + "_" + self.model_name + "_rescaled.csv"
            recall_file = 'Precision_Recall/' + "Recall_" + self.data_name + "_" + self.model_name + "_rescaled.csv"
            precision = load_csv(precision_file)[:, 1:]
            recall = load_csv(recall_file)[:, 1:]

            accuracy_file = '--Masked_Acc_dir' + self.data_name + "_" + self.model_name + "_0_10_20_30_40_50_60_70_80_90_100_percentSal_rescaled.csv"
            accuracy = load_csv(accuracy_file)[:, 1:]

            if feature_precision_recall:
                precision_file_feature = 'Precision_Recall/' + "Precision_" + self.data_name + "_" + \
                                         self.model_name + "_Feature_rescaled.csv"
                recall_file_feature = 'Precision_Recall/' + "/Recall_" + self.data_name + "_" + self.model_name + "_Feature_rescaled.csv"
                feature_precision = load_csv(precision_file_feature)[:, 1:]
                feature_recall = load_csv(recall_file_feature)[:, 1:]

        if time_precision_recall:
            precision_file_time = 'Precision_Recall/' + "Precision_" + self.data_name + "_" + self.model_name + "_Time_rescaled.csv"
            recall_file_time = 'Precision_Recall/' + "/Recall_" + self.data_name + "_" + self.model_name + "_Time_rescaled.csv"
            time_precision = load_csv(precision_file_time)[:, 1:]
            time_recall = load_csv(recall_file_time)[:, 1:]

        for i in range(len(self.saliency_methods)):
            precision_row = []
            recall_row = []

            time_precision_row = []
            time_recall_row = []

            feature_precision_row = []
            feature_recall_row = []

            a = []
            b = 0.1
            for j in range(precision.shape[1]):
                if not pd.isna(precision[i, j]):
                    precision_row.append(precision[i, j])
                    recall_row.append(recall[i, j])

                    if time_precision_recall:
                        time_precision_row.append(time_precision[i, j])
                        time_recall_row.append(time_recall[i, j])

                    if feature_precision_recall:
                        feature_precision_row.append(feature_precision[i, j])
                        feature_recall_row.append(feature_recall[i, j])

                    a.append(b)
                    b += 0.1

            aup = np.trapz(precision_row, x=a)
            aur = np.trapz(recall_row, x=a)
            index_ = np.argsort(recall_row)
            precision_row = np.array(precision_row)
            recall_row = np.array(recall_row)
            aupr = np.trapz(precision_row[index_], x=recall_row[index_])
            aupr_table.append([aupr, self.model_name, self.saliency_methods[i], self.data_name])
            precision_table.append([aup, self.model_name, self.saliency_methods[i], self.data_name])
            recall_table.append([aur, self.model_name, self.saliency_methods[i], self.data_name])

            if time_precision_recall:
                time_aup = np.trapz(time_precision_row, x=a)
                time_aur = np.trapz(time_recall_row, x=a)
                index_ = np.argsort(time_recall_row)
                time_precision_row = np.array(time_precision_row)
                time_recall_row = np.array(time_recall_row)
                time_aupr = np.trapz(time_precision_row[index_], x=time_recall_row[index_])
                time_aupr_table.append([time_aupr, self.model_name, self.saliency_methods[i], self.data_name])
                time_precision_table.append([time_aup, self.model_name, self.saliency_methods[i], self.data_name])
                time_recall_table.append([time_aur, self.model_name, self.saliency_methods[i], self.data_name])

            if feature_precision_recall:
                feature_aup = np.trapz(feature_precision_row, x=a)
                feature_aur = np.trapz(feature_recall_row, x=a)
                index_ = np.argsort(feature_recall_row)
                feature_precision_row = np.array(feature_precision_row)
                feature_recall_row = np.array(feature_recall_row)
                feature_aupr = np.trapz(feature_precision_row[index_], x=feature_recall_row[index_])
                feature_aupr_table.append([feature_aupr, self.model_name, self.saliency_methods[i], self.data_name])
                feature_precision_table.append([feature_aup, self.model_name, self.saliency_methods[i], self.data_name])
                feature_recall_table.append([feature_aur, self.model_name, self.saliency_methods[i], self.data_name])

            accuracy_row = []
            a = []
            b = 0.1
            for j in range(accuracy.shape[1]):
                if not pd.isna(accuracy[i, j]):
                    accuracy_row.append(accuracy[i, j])
                    a.append(b)
                    b += 0.1
            auc = np.trapz(accuracy_row, x=a)
            auc_table.append([auc, self.model_name, self.saliency_methods[i], self.data_name])
            summary_table.append([self.data_name, self.saliency_methods[i], aup, aur, aupr, auc])

        precision_table = np.array(precision_table)
        recall_table = np.array(recall_table)
        aupr_table = np.array(aupr_table)
        auc_table = np.array(auc_table)
        summary_table = np.array(summary_table)

        #save_into_csv(precision_table, 'Accuracy_Metrics/' + self.model_name + "_AUP_rescaled.csv",
         #                    col=precision_table_col)
        #save_into_csv(recall_table, 'Accuracy_Metrics/' + self.model_name + "_AUR_rescaled.csv",
                             #col=recall_table_col)
        #save_into_csv(aupr_table, 'Accuracy_Metrics/' + self.model_name + "_AUPR.csv", col=aupr_table_col)
        #save_into_csv(auc_table, 'Accuracy_Metrics/' + self.model_name + "_AUC.csv", col=auc_table_col)
        #save_into_csv(summary_table, 'Accuracy_Metrics/' + self.model_name + "_preformance_summary.csv",
                             #col=summary_table_col)

        if time_precision_recall:
            time_precision_table = np.array(time_precision_table)
            time_recall_table = np.array(time_recall_table)
            time_aupr_table = np.array(time_aupr_table)

            save_into_csv(time_precision_table,
                                 'Accuracy_Metrics/' + self.model_name + "_AUP_Time_rescaled.csv",
                                 col=precision_table_col)
            save_into_csv(time_recall_table,
                                 'Accuracy_Metrics/' + self.model_name + "_AUR_Time_rescaled.csv",
                                 col=recall_table_col)
            save_into_csv(time_aupr_table, '/Accuracy_Metrics/' + self.model_name + "_AUPR_Time_rescaled.csv",
                                 col=aupr_table_col)

        if feature_precision_recall:
            feature_precision_table = np.array(feature_precision_table)
            feature_recall_table = np.array(feature_recall_table)
            feature_aupr_table = np.array(feature_aupr_table)

            save_into_csv(feature_precision_table,
                                 'Accuracy_Metrics/' + self.model_name + "_AUP_Feature_rescaled.csv",
                                 col=precision_table_col)
            save_into_csv(feature_recall_table,
                                 'Accuracy_Metrics/' + self.model_name + "_AUR_Feature_rescaled.csv",
                                 col=recall_table_col)
            save_into_csv(feature_aupr_table,
                                 'Accuracy_Metrics/' + self.model_name + "_AUPR_Feature_rescaled.csv",
                                 col=aupr_table_col)

        print("Loop finished")

        return summary_table
