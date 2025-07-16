import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import datetime
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets
from tqdm.auto import tqdm



class EvoModel(nn.Module):
    def __init__(self):
        super(EvoModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )
    
    def init_weights(self):
        for param in self.parameters():
            param.data = torch.randn_like(param.data)
        return self
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten the input
        x = x.view(x.size(0), -1)
        return self.layers(x)


class MnistDataset(Dataset):
    def __init__(self, train: bool = True):
        self.train = train
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.dataset = datasets.MNIST(
            root="data",
            train=train,
            download=True,
            transform=self.transform,
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


class EvolutionaryTrainer:
    def __init__(
            self,
            base_model: nn.Module,
            population_size: int = 10,
            apply_mutations: float = 0.5,
            mutation_rate: float = 0.01,
            mutation_magnitude: float = 1.0,
            batch_size: int = 128,
            # Scheduler parameters
            mutation_scheduler: str = "linear",  # "linear", "exponential", "cosine", "none"
            mutation_decay_factor: float = 0.95,  # For exponential decay
            min_mutation_magnitude: float = 0.01,  # Minimum value for mutation magnitude
    ):
        self.population_size = population_size  # Store population size
        self.current_ranking = {}
        self.population = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for i in range(population_size):
            self.population[i] = copy.deepcopy(base_model).init_weights().eval().to(self.device)

        self.apply_mutations = apply_mutations
        self.mutation_rate = mutation_rate
        self.initial_mutation_magnitude = mutation_magnitude
        self.mutation_magnitude = mutation_magnitude
        self.min_mutation_magnitude = min_mutation_magnitude
        self.mutation_scheduler = mutation_scheduler
        self.mutation_decay_factor = mutation_decay_factor
        self.loss_fn = nn.CrossEntropyLoss()
        self.batch_size = batch_size
        self.train_dataset = MnistDataset(train=True)
        self.test_loader = DataLoader(MnistDataset(train=False), batch_size=batch_size, shuffle=False)

    def _update_mutation_magnitude(self, epoch: int, total_epochs: int):
        """Update mutation magnitude based on the selected scheduler."""
        if self.mutation_scheduler == "linear":
            # Linear decay from initial to minimum
            decay_progress = epoch / total_epochs
            self.mutation_magnitude = self.initial_mutation_magnitude * (1 - decay_progress) + self.min_mutation_magnitude * decay_progress
        elif self.mutation_scheduler == "exponential":
            # Exponential decay
            self.mutation_magnitude = max(
                self.min_mutation_magnitude,
                self.initial_mutation_magnitude * (self.mutation_decay_factor ** epoch)
            )
        elif self.mutation_scheduler == "cosine":
            # Cosine annealing
            decay_progress = epoch / total_epochs
            self.mutation_magnitude = self.min_mutation_magnitude + (self.initial_mutation_magnitude - self.min_mutation_magnitude) * (1 + np.cos(np.pi * decay_progress)) / 2
        elif self.mutation_scheduler == "none":
            # No scheduling, keep initial value
            pass
        else:
            raise ValueError(f"Unknown mutation scheduler: {self.mutation_scheduler}")
        
        # Ensure we don't go below minimum
        self.mutation_magnitude = max(self.min_mutation_magnitude, self.mutation_magnitude)

    @torch.no_grad()
    def _mutate(self, model):
        for param in model.parameters():
            if np.random.random() < self.mutation_rate:
                param.data += torch.randn_like(param.data) * self.mutation_magnitude

    @torch.no_grad()
    def _slerp(self, a: nn.Module, b: nn.Module, t: float) -> nn.Module:
        new_model = copy.deepcopy(a)
        for param_new, param_a, param_b in zip(new_model.parameters(), a.parameters(), b.parameters()):
            param_new.data = param_a.data * (1 - t) + param_b.data * t
        return new_model

    @torch.no_grad()
    def _oracle(self, model: nn.Module) -> float:
        loader = Subset(self.train_dataset, np.random.choice(len(self.train_dataset), size=self.batch_size * 10, replace=False))
        loader = DataLoader(loader, batch_size=self.batch_size, shuffle=True)
        #model.to(self.device)
        total_loss = 0.0
        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            pred = model(x)
            loss = self.loss_fn(pred, y)
            total_loss += loss.item()
        #model.cpu()
        return total_loss / len(loader)

    @torch.no_grad()
    def _accuracy(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        preds = logits.argmax(dim=-1)
        return (preds == labels).float().mean().item()

    @torch.no_grad()
    def _evaluate(self) -> float:
        # Get the best model (index 0 after culling)
        best_model = self.population[0]
        #best_model.to(self.device)
        best_accuracy = 0.0
        for x, y in self.test_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            pred = best_model(x)
            accuracy = self._accuracy(pred, y)
            best_accuracy += accuracy
        best_accuracy /= len(self.test_loader)
        #best_model.cpu()
        return best_accuracy

    @torch.no_grad()
    def _cull(self, population: dict[int, nn.Module]) -> dict[int, nn.Module]:
        new_population = {}
        # Get the population indices in order of fitness (best to worst)
        ranked_indices = list(self.current_ranking.keys())
        for i in range(self.population_size):
            new_population[i] = population[ranked_indices[i]]
        return new_population

    @torch.no_grad()
    def train(self, epochs: int = 100):
        # Create log file with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"evolutionary_training_{timestamp}.log"
        
        # Write header to log file
        with open(log_filename, 'w') as log_file:
            log_file.write(f"Evolutionary Neural Network Training Log\n")
            log_file.write(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"Population Size: {self.population_size}\n")
            log_file.write(f"Epochs: {epochs}\n")
            log_file.write(f"Initial Mutation Rate: {self.mutation_rate}\n")
            log_file.write(f"Initial Mutation Magnitude: {self.initial_mutation_magnitude}\n")
            log_file.write(f"Mutation Scheduler: {self.mutation_scheduler}\n")
            log_file.write(f"Min Mutation Magnitude: {self.min_mutation_magnitude}\n")
            if self.mutation_scheduler == "exponential":
                log_file.write(f"Mutation Decay Factor: {self.mutation_decay_factor}\n")
            log_file.write("=" * 100 + "\n\n")
        
        pbar = tqdm(range(epochs), desc="Epochs")
        for epoch in pbar:
            # Update mutation magnitude based on scheduler
            self._update_mutation_magnitude(epoch, epochs)
            
            # Apply changes
            for i, model in tqdm(self.population.items(), desc="Applying changes", leave=False):
                if np.random.random() < self.apply_mutations:
                    self._mutate(model)

            # Slerp population number of new models
            for i in tqdm(range(self.population_size), desc="Slerping population", leave=False):
                random_idx_1 = np.random.randint(0, self.population_size)
                random_idx_2 = np.random.randint(0, self.population_size)
                self.population[i + self.population_size] = self._slerp(
                    self.population[random_idx_1],
                    self.population[random_idx_2],
                    np.random.random(),
                )

            # Evaluate
            for i, model in tqdm(self.population.items(), desc="Oracle", leave=False):
                self.current_ranking[i] = self._oracle(model)

            # Sort population
            self.current_ranking = dict(sorted(self.current_ranking.items(), key=lambda x: x[1]))

            # Cull population
            self.population = self._cull(self.population)

            # Evaluate the best performer
            best_accuracy = self._evaluate()
            pbar.set_postfix({"best_accuracy": best_accuracy, "mutation_mag": f"{self.mutation_magnitude:.4f}"})
            
            # Pretty print top 10 ranking on single line
            top_10 = list(self.current_ranking.items())[:10]
            ranking_str = " | ".join([f"#{rank}: ID{model_id}({fitness:.4f})" for rank, (model_id, fitness) in enumerate(top_10, 1)])
            epoch_log = f"📊 Epoch {epoch + 1} Top 10: {ranking_str} | Mutation Mag: {self.mutation_magnitude:.4f}"
            print(epoch_log)
            
            # Log to file
            with open(log_filename, 'a') as log_file:
                timestamp = datetime.datetime.now().strftime('%H:%M:%S')
                log_file.write(f"[{timestamp}] Epoch {epoch + 1:3d} | Accuracy: {best_accuracy:.6f} | Mut.Mag: {self.mutation_magnitude:.4f} | {ranking_str}\n")

            torch.cuda.empty_cache()
        
        # Write final summary to log
        final_accuracy = self._evaluate()
        with open(log_filename, 'a') as log_file:
            log_file.write("\n" + "=" * 100 + "\n")
            log_file.write(f"Training Completed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"Final Best Accuracy: {final_accuracy:.6f}\n")
            log_file.write(f"Final Mutation Magnitude: {self.mutation_magnitude:.6f}\n")
            log_file.write(f"Total Epochs: {epochs}\n")
            log_file.write("=" * 100 + "\n")
        
        print(f"\n🎯 Training completed! Log saved to: {log_filename}")
        print(f"🏆 Final best accuracy: {final_accuracy:.6f}")
        print(f"📉 Final mutation magnitude: {self.mutation_magnitude:.6f}")


if __name__ == "__main__":
    # Initialize model
    base_model = EvoModel()
    
    # Create trainer
    trainer = EvolutionaryTrainer(
        base_model=base_model,
        population_size=1000,
        apply_mutations=0.5,
        mutation_rate=0.01,
        mutation_magnitude=10.0,
        batch_size=1024,
        mutation_scheduler="linear",
        mutation_decay_factor=0.95,
        min_mutation_magnitude=0.00001,
    )
    
    # Train
    trainer.train(epochs=50)
