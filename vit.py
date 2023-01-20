import requests
import torch
import torchvision.datasets
import torch.utils.data
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import ImageGPTImageProcessor, ImageGPTModel, ImageGPTFeatureExtractor, ViTFeatureExtractor, ViTModel, ViTForImageClassification
from sklearn.manifold import TSNE
import numpy as np
from datasets import load_dataset


np.set_printoptions(precision=4, suppress=True)
torch.set_printoptions(precision=4, sci_mode=False)


def draw_tsne_visualization(X, n_train, n_test_nominal, n_test_anomalous, points_shown, dim=2):
    print(X.shape)
    X_tsne_embedded = TSNE(n_components=dim, perplexity=50, n_iter=4000).fit_transform(X)
    shown_train_points = min(n_train, points_shown)
    shown_test_nominal_points = min(n_test_nominal, points_shown)
    shown_test_anomalous_points = min(n_test_anomalous, points_shown)
    plt.scatter(
        np.concatenate((X_tsne_embedded[0:shown_train_points, 0], X_tsne_embedded[n_train:n_train + shown_test_nominal_points, 0], X_tsne_embedded[n_train + n_test_nominal:n_train + n_test_nominal + shown_test_anomalous_points, 0])),
        np.concatenate((X_tsne_embedded[0:shown_train_points, 1], X_tsne_embedded[n_train:n_train + shown_test_nominal_points, 1], X_tsne_embedded[n_train + n_test_nominal:n_train + n_test_nominal + shown_test_anomalous_points, 1])),
        c=['blue' for i in range(shown_train_points)] + ['green' for i in range(shown_test_nominal_points)] + ['red' for i in range(shown_test_anomalous_points)]
    )
    plt.show()

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# train_dataset = torchvision.datasets.CIFAR10(
#     root='./data',
#     train=True,
#     download=True,
#     transform=torchvision.transforms.Compose([
#         # torchvision.transforms.Resize((224, 224)),
#         torchvision.transforms.ToTensor()
#     ])
# )
# train_dataloader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_dataset, list(range(10_000))), batch_size=1)
#
# test_dataset = torchvision.datasets.CIFAR10(
#     root='./data',
#     train=False,
#     download=True,
#     transform=torchvision.transforms.Compose([
#         # torchvision.transforms.Resize((224, 224)),
#         torchvision.transforms.ToTensor()
#     ])
# )
# test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

train_data = load_dataset('cifar10', split='train')
train_data_loader = torch.utils.data.DataLoader(train_data.with_format('torch'), batch_size=2)
test_data = load_dataset('cifar10', split='test')
test_data_loader = torch.utils.data.DataLoader(test_data.with_format('torch'), batch_size=2)

print(train_data[0])
plt.imshow(train_data[0]['img'])
plt.show()

image_feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# inputs = image_feature_extractor(images=train_data[0:1]['img'], return_tensors="pt")
# print(inputs.pixel_values.shape)
#
# outputs = model(inputs.pixel_values.to(device), output_hidden_states=True)
# hidden_states = outputs.hidden_states
#
# print(len(hidden_states))
# print(hidden_states[0].shape)
# print(hidden_states[len(hidden_states)//2].mean(dim=1).shape)



for normal_class in range(10):
    print(f"Normal class: {normal_class}")
    x_data = np.zeros((0, 768))
    y_data = np.zeros(0)

    count_normal = 0
    count_anomalous = 0

    for i in tqdm(range(len(train_data))):
        if train_data[i]['label'] == normal_class:
            if count_normal >= 9_500:
                continue
            count_normal += 1
        else:
            if count_anomalous >= 275:
                continue
            count_anomalous += 1

        inputs = image_feature_extractor(train_data[i]['img'], return_tensors="pt")
        outputs = model(inputs.pixel_values.to(device), output_hidden_states=True)
        representation = outputs.hidden_states[12].mean(dim=1)

        x_data = np.append(x_data, representation.detach().cpu().numpy(), axis=0)
        y_data = np.append(y_data, np.asarray([int(train_data[i]['label'] != normal_class)]), axis=0)

    np.savez_compressed(f'CIFAR10_{normal_class}_vit-b-16-last-layer-avg.npz', X=x_data, y=y_data)