import torch
import torch.nn as nn


class DiscriminatorEnsemble(nn.Module):
    def __init__(self,
                 observation_dim,
                 observation_horizon,
                 num_classes,
                 device,
                 shape=[1024, 512],
                 ensemble_size=5,
                 eps=1e-7,
                 incremental_input=False,
                 ):
        super(DiscriminatorEnsemble, self).__init__()
        self.observation_dim = observation_dim
        self.observation_horizon = observation_horizon
        self.input_dim = observation_dim * observation_horizon
        self.device = device
        self.num_classes = num_classes
        self.shape = shape
        self.ensemble_size = ensemble_size
        self.eps = eps
        self.incremental_input = incremental_input
        
        self.ensemble = nn.ModuleList(
            [
                Discriminator(
                    observation_dim,
                    observation_horizon,
                    num_classes,
                    device,
                    shape,
                    eps,
                    incremental_input,
                    )
                for _ in range(ensemble_size)
                ]
        )
        
    def forward(self, x):
        # bootstrapping
        indices = []
        out = []
        for discriminator in self.ensemble:
            idx = torch.randint(0, x.size(0), (x.size(0),), device=self.device)
            indices.append(idx)
            out.append(discriminator.architecture(x[idx]))
        return out, indices
    
    def compute_dis_skill_reward(self, observation_buf, style_selector):
        with torch.no_grad():
            self.eval()
            logp_ensemble = []
            for discriminator in self.ensemble:
                discriminator.eval()
                logp = discriminator.predict_logp(observation_buf, style_selector)
                logp_ensemble.append(logp)
                discriminator.train()
            logp_avg = torch.log(torch.exp(torch.cat(logp_ensemble, dim=1)).mean(dim=1))
            skill_reward = logp_avg - torch.log(torch.tensor(1 / self.num_classes, device=self.device))
            self.train()
        return torch.clip(skill_reward, min=0.0)

    def compute_dis_disdain_reward(self, observation_buf):
        with torch.no_grad():
            self.eval()
            entropy_ensemble = []
            probs_ensemble = []
            for discriminator in self.ensemble:
                discriminator.eval()
                entropy_ensemble.append(discriminator.predict_entropy(observation_buf).unsqueeze(1))
                logits = discriminator.predict_logits(observation_buf).unsqueeze(1)
                probs = nn.functional.softmax(logits, dim=2)
                probs_ensemble.append(probs)
                discriminator.train()
            # mean of the entropy
            entropy_avg = torch.cat(entropy_ensemble, dim=1).mean(dim=1)
            # entropy of the mean
            probs_avg = torch.cat(probs_ensemble, dim=1).mean(dim=1)
            entropy = (-probs_avg * torch.log(probs_avg + self.eps)).sum(dim=1)            
            # DISDAIN reward
            disdain_reward = entropy - entropy_avg
            self.train()
        return disdain_reward

class Discriminator(nn.Module):
    def __init__(self,
                 observation_dim,
                 observation_horizon,
                 num_classes,
                 device,
                 shape=[1024, 512],
                 eps=1e-7,
                 incremental_input=False,
                 ):
        super(Discriminator, self).__init__()
        self.observation_dim = observation_dim
        self.observation_horizon = observation_horizon
        self.input_dim = observation_dim * observation_horizon
        self.device = device
        self.num_classes = num_classes
        self.shape = shape
        self.eps = eps
        self.incremental_input = incremental_input

        discriminator_layers = []
        curr_in_dim = self.input_dim
        discriminator_layers.append(nn.BatchNorm1d(curr_in_dim))
        for hidden_dim in self.shape:
            discriminator_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            discriminator_layers.append(nn.ReLU())
            curr_in_dim = hidden_dim
        discriminator_layers.append(nn.Linear(self.shape[-1], self.num_classes))
        self.architecture = nn.Sequential(*discriminator_layers).to(self.device)
        self.architecture.train()

    def forward(self, x):
        return self.architecture(x)

    def predict_logits(self, observation_buf):
        with torch.no_grad():
            self.eval()
            if self.incremental_input:
                logits = self.architecture((observation_buf - observation_buf[:, :1, :]).flatten(1, 2))
            else:
                logits = self.architecture((observation_buf).flatten(1, 2))
            self.train()
        return logits

    def predict_logp(self, observation_buf, style_selector):
        with torch.no_grad():
            self.eval()
            logits = self.predict_logits(observation_buf)
            logp = torch.gather(nn.functional.log_softmax(logits, dim=1), dim=1, index=style_selector.unsqueeze(-1))
            self.train()
        return logp
    
    def predict_entropy(self, observation_buf):
        with torch.no_grad():
            self.eval()
            logits = self.predict_logits(observation_buf)
            probs = nn.functional.softmax(logits, dim=1)
            entropy = (-probs * torch.log(probs + self.eps)).sum(dim=1)
            self.train()
        return entropy
