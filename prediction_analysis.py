import numpy as np
import matplotlib.pyplot as plt
from matplotlib.container import BarContainer
import pandas as pd
import torch
import os
from tqdm import tqdm
import json


class PredictionVisualization:
    def __init__(self, labels_pred, labels_true, sort_after: str = 'race'):
        assert sort_after == 'gender' or sort_after == 'race', f"Choose 'gender' or 'race' as sort_after"
        self.sort_after = sort_after
        self.gender_names = ['Male', 'Female']
        self.race_names = ['Black', 'East Asian', 'Indian', 'Latino_Hispanic', 'Middle Eastern', 'Southeast Asian', 'White']
        self.class_names = self.get_class_names()
        self.pred = self.get_labels_dict(labels_pred, labels_true)
        self.true = self.get_labels_dict(labels_true, labels_true)
        self.df = pd.DataFrame(list(zip(self.pred['gender_name'], self.pred['race_name'],
                                        self.true['gender_name'], self.true['race_name'])),
                               columns=['gender_pred', 'race_pred', 'gender_true', 'race_true'])

    def get_labels_dict(self, labels, labels_true):
        assert labels[0].size in [2, 7, 9], f'Unexpected prediction label size {labels[0].size}'
        if labels[0].size == 2:
            # if race predictions are missing use true race labels
            race_labels, race_name = self.get_labels(labels_true, mode='race')
            gender_labels, gender_name = self.get_labels(labels, mode='gender')
        elif labels[0].size == 7:
            # if gender predictions are missing use true gender labels
            race_labels, race_name = self.get_labels(labels, mode='race')
            gender_labels, gender_name = self.get_labels(labels_true, mode='gender')
        else:
            race_labels, race_name = self.get_labels(labels, mode='race')
            gender_labels, gender_name = self.get_labels(labels, mode='gender')
        if self.sort_after == 'race':
            class_labels = len(self.gender_names) * race_labels + gender_labels
        else:
            class_labels = len(self.race_names) * gender_labels + race_labels
        return {'gender_labels': gender_labels, 'race_labels': race_labels,
                'gender_name': gender_name, 'race_name': race_name,
                'class_labels': class_labels}

    def get_labels(self, labels_one_hot: np.array, mode):
        if mode == 'gender':
            gender_labels = np.argmax(labels_one_hot[:, :2], axis=1)
            gender_name = [self.gender_names[idx] for idx in gender_labels]
            return gender_labels, gender_name
        elif mode == 'race':
            race_labels = np.argmax(labels_one_hot[:, -7:], axis=1)
            race_name = [self.race_names[idx] for idx in race_labels]
            return race_labels, race_name
        else:
            raise ValueError(f'Wrong mode {mode}')

    def get_class_names(self):
        if self.sort_after == 'race':
            class_names = [f'{r_n} {g_n}' for r_n in self.race_names for g_n in self.gender_names]
        else:
            class_names = [f'{r_n} {g_n}' for g_n in self.gender_names for r_n in self.race_names]
        return class_names

    def plot_histogram(self, fn:str = None):
        labels_pred, labels_true, classnames = self.pred['class_labels'], self.true['class_labels'], self.class_names
        K = len(classnames)

        # 2d histogram
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.subplots_adjust(left=0.32, bottom=0.32)
        ax.set_title('Predicted vs true labels')
        ax.hist2d(labels_true, labels_pred, bins=K)
        ax.set_xlabel('True labels')
        ax.set_xticks([0.5 + k * ((K-1)/K) for k in range(K)], classnames, rotation=90)
        ax.set_ylabel('Predicted labels')
        ax.set_yticks([0.5 + k * ((K-1)/K) for k in range(K)], classnames)
        if fn is None:
            plt.show()
        else:
            plot_name = f'figs/histo_2d_{fn}'
            plt.savefig(plot_name)
        plt.close(fig)

        # 1d histogram
        correct_labels = []
        incorrect_labels = []
        for label_true, label_pred in zip(labels_true, labels_pred):
            if label_true == label_pred:
                correct_labels.append(label_true)
            else:
                incorrect_labels.append(label_true)

        # Creating histogram
        fig, ax = plt.subplots(2, 1, figsize=(10, 10))
        plt.subplots_adjust(hspace=1, top=0.95, bottom=0.27)
        ax[0].hist(correct_labels, color='green', bins=K)
        ax[0].set_title('Correctly classified labels')
        ax[0].set_xticks([0.5 + k * ((K-1)/K) for k in range(K)], classnames, rotation=60, ha='right')
        ax[1].hist(incorrect_labels, color='red', bins=K)
        ax[1].set_title('Incorrectly classified labels')
        ax[1].set_xticks([0.5 + k * ((K-1)/K) for k in range(K)], classnames, rotation=60, ha='right')
        for a in ax:
            for container in a.containers:
                a.bar_label(container)
        if fn is None:
            plt.show()
        else:
            plot_name = f'figs/histo_1d_{fn}'
            plt.savefig(plot_name)
        plt.close(fig)

    def plot_gender_acc(self, normalize: bool = False, fn:str = None):
        gender_and_race_correct = self.get_gender_and_race_values(gender=True, race=True)
        gender_correct = self.get_gender_and_race_values(gender=True, race=False)
        race_correct = self.get_gender_and_race_values(gender=False, race=True)
        none_correct = self.get_gender_and_race_values(gender=False, race=False)
        race_and_none = {key: race_correct[key] + none_correct[key] for key in race_correct}
        gender_and_none = {key: gender_correct[key] + none_correct[key] for key in gender_correct}
        gender = {key: [d[key] for d in [gender_and_race_correct, gender_correct, race_and_none]] for key in
                  gender_and_race_correct}
        race = {key: [d[key] for d in [gender_and_race_correct, race_correct, gender_and_none]] for key in
                gender_and_race_correct}

        df_gender = pd.DataFrame.from_dict(gender, orient='index',
                                           columns=['Gender and Race correct', 'Gender correct', 'Incorrect'])
        df_race = pd.DataFrame.from_dict(race, orient='index',
                                         columns=['Gender and Race correct', 'Race correct', 'Incorrect'])

        if normalize:
            df_gender = df_gender.div(df_gender.sum(axis=1) * 0.01, axis=0).round(1)
            df_race = df_race.div(df_race.sum(axis=1) * 0.01, axis=0).round(1)

            fig, axes = plt.subplots(1, 2, figsize=(12, 7))
            title = f'Gender and Race classification visualization\nSize of dataset: {self.df.shape[0]}'
            plt.suptitle(title)
            plt.subplots_adjust(wspace=0.5, left=0.16, right=0.98)
            for ax, df, name in zip(axes, [df_gender, df_race], ['Gender', 'Race']):
                ax.set_title(name)
                df.plot.barh(ax=ax, stacked=True, color=['limegreen', 'gold', 'orangered'])
                mean = [df.values[:, :2].sum(axis=1).mean().round(1), df.values[:, :1].sum(axis=1).mean().round(1)]
                std = [df.values[:, :2].sum(axis=1).std().round(1), df.values[:, :1].sum(axis=1).std().round(1)]
                hbars = ax.barh([-1, -2], mean, height=.6, xerr=std, color='deepskyblue')
                ax.bar_label(hbars, labels=[f'±{std}' for std in std])
                ax.set_yticks(np.arange(-2, len(self.class_names)), labels=['Gender and Race', name, *self.class_names])
                for container in ax.containers:
                    if isinstance(container, BarContainer):
                        if any(container.datavalues):
                            ax.bar_label(container, label_type='center')
                ax.set_ylim(bottom=-5)

            mean = df_gender.values[:, :1].sum(axis=1).mean()
            std = df_gender.values[:, :1].sum(axis=1).std()
            attention_scores = 2 ** ((mean - df_gender.values[:, :1].sum(axis=1)) / (2 * std))
            attention_dict = {key: value for key, value in zip(self.class_names, attention_scores)}
            with open(f"attention_scores/attention_scores_{fn}.json", "w") as f:
                json.dump(attention_dict, f)

        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            title = f'Gender and Race classification visualization\nSize of dataset: {self.df.shape[0]}'
            plt.suptitle(title)
            plt.subplots_adjust(wspace=0.5, left=0.16, right=0.98)
            for ax, df, name in zip(axes, [df_gender, df_race], ['Gender', 'Race']):
                ax.set_title(name)
                df.plot.barh(ax=ax, stacked=True, color=['limegreen', 'gold', 'orangered'])
                for container in ax.containers:
                    if any(container.datavalues):
                        ax.bar_label(container, label_type='center')
                ax.set_ylim(bottom=-3)

        if fn is None:
            plt.show()
        else:
            plot_name = f'figs/plot_normalize_{normalize}_{fn}'
            plt.savefig(plot_name)
        plt.close(fig)

    def get_gender_and_race_values(self, gender: bool, race: bool):
        if self.sort_after == 'race':
            return {f'{r} {g}': len(self.df[((self.df['gender_true'] == self.df['gender_pred']) == gender) &
                                            (self.df['gender_true'] == g) &
                                            ((self.df['race_true'] == self.df['race_pred']) == race) &
                                            (self.df['race_true'] == r)])
                    for r in self.race_names for g in self.gender_names}
        else:
            return {f'{r} {g}': len(self.df[((self.df['gender_true'] == self.df['gender_pred']) == gender) &
                                            (self.df['gender_true'] == g) &
                                            ((self.df['race_true'] == self.df['race_pred']) == race) &
                                            (self.df['race_true'] == r)])
                    for g in self.gender_names for r in self.race_names}


def encode_labels_to_one_hot(labels_from_csv: pd.DataFrame):
    gender_names = ['Male', 'Female']
    race_names = ['Black', 'East Asian', 'Indian', 'Latino_Hispanic', 'Middle Eastern', 'Southeast Asian', 'White']

    gender = labels_from_csv.get('gender')
    race = labels_from_csv.get('race')

    gender_idx = [gender_names.index(g) for g in gender]
    race_idx = [race_names.index(r) for r in race]

    labels_true_gender_one_hot = pd.get_dummies(gender_idx)
    labels_true_race_one_hot = pd.get_dummies(race_idx)
    labels_true_one_hot = pd.concat([labels_true_gender_one_hot, labels_true_race_one_hot], axis=1).to_numpy()

    np.save('val_True_gender_race_one_hot', labels_true_one_hot)


def get_sample_data(n_samples=1000, n_genders=2, n_races=7):
    labels_pred_gender_one_hot = pd.get_dummies(np.random.randint(0, n_genders, n_samples))
    labels_pred_race_one_hot = pd.get_dummies(np.random.randint(0, n_races, n_samples))
    labels_pred_one_hot = pd.concat([labels_pred_gender_one_hot, labels_pred_race_one_hot], axis=1).to_numpy()
    labels_true_gender_one_hot = pd.get_dummies(np.random.randint(0, n_genders, n_samples))
    labels_true_race_one_hot = pd.get_dummies(np.random.randint(0, n_races, n_samples))
    labels_true_one_hot = pd.concat([labels_true_gender_one_hot, labels_true_race_one_hot], axis=1).to_numpy()
    return labels_pred_one_hot, labels_true_one_hot


if __name__ == "__main__":
    for fn in tqdm(os.listdir('predictions')):
        if fn.startswith('predictions') or fn.startswith('val_predictions'):
            pass
        else:
            continue
        fn_pred = f"predictions/{fn}"
        labels_pred = np.load(fn_pred, allow_pickle=True)
        if isinstance(labels_pred[0], tuple):
            for i, label in enumerate(labels_pred):
                labels_pred[i] = torch.concat(list((label[1], label[0])), dim=1)

        labels_pred = torch.concat(labels_pred).cpu().data.numpy()

        # annotations_file = 'val_True.csv'
        # labels_from_csv = pd.read_csv(annotations_file)
        # labels_true = encode_labels_to_one_hot(labels_from_csv)

        if labels_pred.shape[0] == 5162:
            labels_true = np.load('groundtruth/test_True_gender_race_one_hot.npy')
        elif labels_pred.shape[0] == 10954:
            labels_true = np.load('groundtruth/test_gender_race_one_hot.npy')
        elif labels_pred.shape[0] == 5031:
            labels_true = np.load('groundtruth/val_True_gender_race_one_hot.npy')
        elif labels_pred.shape[0] == 10841:
            labels_true = np.load('groundtruth/val_gender_race_one_hot.npy')
        else:
            raise ValueError(f'Size of Prediction {labels_pred.shape[0]} unknown')

        # create sample data
        # labels_pred, labels_true = get_sample_data()

        pred_vis = PredictionVisualization(labels_pred, labels_true, sort_after='race')

        fn = os.path.splitext(fn)[0]
        # pred_vis.plot_gender_acc(normalize=True)
        pred_vis.plot_gender_acc(normalize=True, fn=fn)
        pred_vis.plot_gender_acc(normalize=False, fn=fn)
        pred_vis.plot_histogram(fn=fn)

