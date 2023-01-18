import requests
import torch
import torchvision.datasets
import torch.utils.data
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import ImageGPTImageProcessor, ImageGPTModel
from sklearn.manifold import TSNE
import numpy as np


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

train_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=torchvision.transforms.Compose([
        # torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor()
    ])
)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=torchvision.transforms.Compose([
        # torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor()
    ])
)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

image_processor = ImageGPTImageProcessor.from_pretrained("openai/imagegpt-small")
model = ImageGPTModel.from_pretrained('openai/imagegpt-small')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

inputs = image_processor(images=train_dataset[1][0], return_tensors="pt").to(device)
outputs = model(**inputs, output_hidden_states=True)
last_hidden_states = outputs.last_hidden_state


x_data = np.zeros((0, 512))
y_data = np.zeros((0, 1))

for x, y in tqdm(train_dataloader):
    inputs = image_processor(images=x[0], return_tensors="pt").to(device)
    outputs = model(**inputs, output_hidden_states=True)
    representation = outputs.hidden_states[len(outputs.hidden_states)//2].mean(dim=1)

    x_data = np.append(x_data, representation.detach().cpu().numpy(), axis=0)
    print((y == 0).detach().cpu().numpy())
    y_data = np.append(y_data, (y == 0).detach().cpu().numpy(), axis=0)


# draw_tsne_visualization(np.append(nominal_data, anomalous_data, axis=0), 200, 0, 200, 200)

representation = outputs.hidden_states[len(outputs.hidden_states)//2].mean(dim=1)

print(representation.size())
print(representation)