# pictureSorter.py

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

class PictureSorter:
    def __init__(self, device=None):
        """
        Initialize the PictureSorter with a pre-trained VGG16 model.

        Args:
            device (str, optional): Device to run the model on ('cpu' or 'cuda'). Defaults to 'cuda' if available.
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        self.preprocess = self._define_preprocess()

    def _load_model(self):
        """Load and prepare the pre-trained VGG16 model."""
        model = models.vgg16(pretrained=True)
        # Remove the last layer to get feature vectors
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        model.to(self.device)
        model.eval()
        return model

    def _define_preprocess(self):
        """Define the image preprocessing pipeline."""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def extract_features(self, image_path):
        """
        Extract feature vector from an image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            np.ndarray: Flattened feature vector.
        """
        try:
            img = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return None

        img_t = self.preprocess(img)
        batch_t = torch.unsqueeze(img_t, 0).to(self.device)

        with torch.no_grad():
            features = self.model(batch_t)
        return features.cpu().numpy().flatten()

    def sort_images(self, image_dir, output_dir, n_clusters=5):
        """
        Sort images into clusters based on extracted features.

        Args:
            image_dir (str): Directory containing images to sort.
            output_dir (str): Directory where sorted images will be saved.
            n_clusters (int, optional): Number of clusters for KMeans. Defaults to 5.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Get list of all image paths
        image_paths = [
            os.path.join(image_dir, fname) for fname in os.listdir(image_dir)
            if fname.lower().endswith(('png', 'jpg', 'jpeg'))
        ]

        if not image_paths:
            print("No images found in the source directory.")
            return

        print("Extracting features from images...")
        features = []
        valid_image_paths = []
        for image_path in image_paths:
            feature = self.extract_features(image_path)
            if feature is not None:
                features.append(feature)
                valid_image_paths.append(image_path)

        if not features:
            print("No valid images to process.")
            return

        features = np.array(features)

        print("Reducing dimensionality with PCA...")
        pca = PCA(n_components=50)
        features_reduced = pca.fit_transform(features)

        print("Clustering images with KMeans...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        labels = kmeans.fit_predict(features_reduced)

        print("Sorting images into clusters...")
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
                image_path = valid_image_paths[idx]
                image_name = os.path.basename(image_path)
                new_path = os.path.join(cluster_dir, image_name)
                shutil.copy(image_path, new_path)

        print(f'Sorted images are saved in {output_dir}')