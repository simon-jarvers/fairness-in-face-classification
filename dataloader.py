import os
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import json


class FaceDataset(Dataset):
    def __init__(self, annotations_file, img_dir, device=None, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.device = device
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = (read_image(img_path)/255).to(device=self.device, non_blocking=True)
        # one-hot-encoding
        label = torch.tensor(int(self.img_labels.iloc[idx, 2] == 'Female'))
        label = torch.nn.functional.one_hot(label, num_classes=2)
        label = label.float().to(device=self.device, non_blocking=True)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def split_dataset(face_dataset: FaceDataset, train_split: float):
    df: pd.DataFrame = face_dataset.img_labels

    df_grouped = df.groupby(['service_test', 'gender', 'race'])

    train_set = []
    for name, group in df_grouped:
        random_set = group.sample(frac=train_split)
        train_set.append(random_set)

    df_train = pd.concat(train_set)
    df_val = pd.concat([df, df_train]).drop_duplicates(keep=False)

    df_train.to_csv('./train.csv', index=False)
    df_val.to_csv('./val.csv', index=False)


def split_based_on_service_test(face_dataset: FaceDataset, service_test = True, fn='test'):
    df: pd.DataFrame = face_dataset.img_labels

    df_grouped = df.groupby(['service_test'])

    for service_test_bool, group in df_grouped:
        group.to_csv(f'./{fn}_{service_test_bool}.csv', index=False)


def create_unbalanced_dataset(face_dataset: FaceDataset, service_test: bool, race: str, fn: str):
    df: pd.DataFrame = face_dataset.img_labels

    df_true = df[df['service_test'] == service_test]
    df_race = df[df['race'] == race]
    df_true_and_one_race = pd.concat([df_true, df_race]).drop_duplicates(keep=False)

    df_true_and_one_race.to_csv(f'./{fn}_{service_test}_{race}.csv', index=False)


def dataset_balance(face_dataset: FaceDataset):
    df: pd.DataFrame = face_dataset.img_labels

    df_true = df[df['service_test']]
    df_false = df[df['service_test'] == False]

    plot_gender_race(df_true, f'Variable service_test is True\nSize of dataset: {df_true.shape[0]}')
    plot_gender_race(df_false, f'Variable service_test is False\nSize of dataset: {df_false.shape[0]}')


def plot_gender_race(df: pd.DataFrame, title: str):
    df_gender = (df.groupby(['gender', 'race'])
                 .size()
                 .unstack(level=-1)
                 .reset_index()
                 )
    df_race = (df.groupby(['race', 'gender'])
                 .size()
                 .unstack(level=-1)
                 .reset_index()
                 )

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plt.suptitle(title)
    plt.subplots_adjust(wspace=0.4)
    for ax, df, name in zip(axes, [df_gender, df_race], ['gender', 'race']):
        df.plot.barh(x=name, ax=ax, stacked=True)
        for container in ax.containers:
            ax.bar_label(container, label_type='center')
        ax.set_ylim(bottom=-1.5)
    plt.show()


def save_images_number(face_dataset: FaceDataset):
    df: pd.DataFrame = face_dataset.img_labels

    df_grouped = df.groupby(['gender', 'race'])

    n_images = {}
    for name, group in df_grouped:
        n_images[f'{name[1]} {name[0]}'] = group.shape[0]

    with open(f"n_images_per_class.json", "w") as f:
        json.dump(n_images, f)


if __name__ == '__main__':
    # training_data = FaceDataset('./fairface_label_train.csv', '.')
    #
    # split_dataset(training_data, 0.875)

    # train_True_White = FaceDataset('./train_True_White.csv', '.')
    # val_True_White = FaceDataset('./val_True_White.csv', '.')
    # test_True_White = FaceDataset('./test_True_White.csv', '.')
    # plot_gender_race(train_True_White.img_labels, f'Training data for unbalanced dataset\n'
    #                                               f'Size of dataset: {train_True_White.img_labels.shape[0]}')
    # plot_gender_race(val_True_White.img_labels, f'Validation data for unbalanced dataset\n'
    #                                               f'Size of dataset: {val_True_White.img_labels.shape[0]}')
    # plot_gender_race(test_True_White.img_labels, f'Test data for unbalanced dataset\n'
    #                                               f'Size of dataset: {test_True_White.img_labels.shape[0]}')

    training_data = FaceDataset('./train.csv', '.')
    save_images_number(training_data)
    dataset_balance(training_data)

    val_data = FaceDataset('./val.csv', '.')
    dataset_balance(val_data)

    test_data = FaceDataset('./fairface_label_val.csv', '.')
    dataset_balance(test_data)

    # create_unbalanced_dataset(training_data, service_test=True, race='White', fn='train')
    # create_unbalanced_dataset(val_data, service_test=True, race='White', fn='val')
    # create_unbalanced_dataset(test_data, service_test=True, race='White', fn='test')
    # split_based_on_service_test(val_data, fn='val')

    # train_dataloader = DataLoader(training_data, batch_size=64, shuffle=False)
    #
    # # Display image and label.
    # train_features, train_labels = next(iter(train_dataloader))
    # print(f"Feature batch shape: {train_features.size()}")
    # for i in range(3):
    #     img = train_features[i].permute(1, 2, 0)
    #     label = train_labels[i]
    #     plt.imshow(img)
    #     plt.title(label)
    #     plt.show()
    #     print(f"Label: {label}")


