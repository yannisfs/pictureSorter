import os
import shutil
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load pre-trained model
model = models.vgg16(pretrained=True)
model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])  # Remove the last layer
model.eval()

# Define transformation
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image_path, model, preprocess):
    img = Image.open(image_path).convert('RGB')
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)
    
    with torch.no_grad():
        features = model(batch_t)
    return features.numpy().flatten()

# Specify the directory containing images
image_dir = 'path/to/image_folder'
output_dir = 'path/to/output_folder'

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get list of all image paths
image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.lower().endswith(('png', 'jpg', 'jpeg'))]

# Load images and extract features
features = np.array([extract_features(image_path, model, preprocess) for image_path in image_paths])

# Reduce dimensionality
pca = PCA(n_components=50)
features_reduced = pca.fit_transform(features)

# Cluster images
kmeans = KMeans(n_clusters=5, random_state=0)
labels = kmeans.fit_predict(features_reduced)

# Sort images within each cluster and save to new folders
for cluster_num in np.unique(labels):
    cluster_indices = np.where(labels == cluster_num)[0]
    cluster_features = features_reduced[cluster_indices]
    cluster_center = kmeans.cluster_centers_[cluster_num]
    distances = np.linalg.norm(cluster_features - cluster_center, axis=1)
    sorted_cluster_indices = cluster_indices[np.argsort(distances)]
    
    # Create a directory for each cluster
    cluster_dir = os.path.join(output_dir, f'cluster_{cluster_num}')
    if not os.path.exists(cluster_dir):
        os.makedirs(cluster_dir)
    
    # Save sorted images to the cluster directory
    for idx in sorted_cluster_indices:
        image_path = image_paths[idx]
        image_name = os.path.basename(image_path)
        new_path = os.path.join(cluster_dir, image_name)
        shutil.copy(image_path, new_path)

print(f'Sorted images are saved in {output_dir}')