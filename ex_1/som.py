import torch
from math import exp
from tqdm import trange
import sys


class SOM:
    def __init__(
        self,
        num_neurons=500,
        initial_lr=0.5,
        lr_decay_time=25,
        initial_radius=2,
        radius_decay_time=25,
        iterations=300,
        random_seed=42,
        shuffle_data=True,
        normalize=True,
    ):
        self.config = {
            "num_neurons": num_neurons,  # Total neurons in the map
            "initial_lr": initial_lr,  # Starting learning rate
            "lr_decay_time": lr_decay_time,  # Learning rate decay factor
            "initial_radius": initial_radius,  # Starting neighborhood radius
            "radius_decay_time": radius_decay_time,  # Radius decay factor
            "iterations": iterations,  # Total training epochs
            "random_seed": random_seed,  # Seed for reproducibility
            "shuffle_data": shuffle_data,  # Shuffle input data
            "normalize": normalize,  # Scale weights to match input data
        }

        if self.config["random_seed"] is not None:
            torch.manual_seed(random_seed)

    def train(self, data):
        num_samples, num_features = data.shape

        # Initialize neuron weights randomly
        self.weights = torch.rand(
            (self.config["num_neurons"], num_features), dtype=torch.double
        )

        # Convert input data to torch tensor
        input_data = torch.from_numpy(data).type(torch.double)

        # Shuffle data if required
        if self.config["shuffle_data"]:
            permutation = torch.randperm(num_samples)
            input_data = input_data[permutation, :]

        # Normalize weights to match the range of input data
        if self.config["normalize"]:
            data_min, data_max = torch.min(input_data), torch.max(input_data)
            self.weights = self.weights * (data_max - data_min) + data_min

        # Training loop
        for epoch in trange(self.config["iterations"], desc="Training SOM"):
            for current_input in input_data:
                distance_vector = (
                    current_input - self.weights
                )  # Distance between input and weights

                # Calculate squared Euclidean distances
                squared_distances = torch.sum(distance_vector**2, dim=1)
                winner_idx = torch.argmin(squared_distances)

                # Compute lateral distances from the winning neuron
                lateral_distances = torch.sum(
                    (self.weights - self.weights[winner_idx, :]) ** 2, dim=1
                )

                # Update learning rate and neighborhood radius
                learning_rate = self.config["initial_lr"] * exp(
                    -epoch / self.config["lr_decay_time"]
                )
                neighborhood_radius = self.config["initial_radius"] * exp(
                    -epoch / self.config["radius_decay_time"]
                )

                # Compute the neighborhood function
                influence = torch.exp(
                    -lateral_distances / (2 * neighborhood_radius**2)
                ).unsqueeze(1)

                # Update weights
                self.weights += learning_rate * influence * distance_vector

    def compute_adjacency(self, data_matrix):
        data_tensor = torch.from_numpy(data_matrix).type(torch.double)
        num_samples, num_features = data_tensor.shape

        # Expand data tensor for pairwise distance computation
        expanded_tensor = data_tensor.unsqueeze(0).repeat(num_samples, 1, 1)
        flat_tensor = data_tensor.unsqueeze(1).repeat(1, num_samples, 1)

        # Calculate Euclidean distances
        distances = torch.sqrt(torch.sum((expanded_tensor - flat_tensor) ** 2, dim=2))
        return distances

    def update_parameters(self, new_params):
        self.config.update(new_params)

    def retrieve_parameters(self):
        return self.config

    def get_current_weights(self):
        return self.weights
