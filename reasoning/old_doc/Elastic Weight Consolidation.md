# Elastic Weight Consolidation

EWC (Elastic Weight Consolidation), we use it to estimate which weights (parameters) are most important for the original task.

Here's a simpler breakdown:

1. Sensitivity Indicator:
When you train a model, some parameters are very critical—small changes to them can lead to large increases in the loss (i.e., poorer performance). Fisher Information quantifies that sensitivity.

2. How It’s Estimated:
In practice, we approximate the importance of each parameter by taking the squared gradient (i.e., the derivative of the loss with respect to that parameter) and averaging it over many data samples. A higher average (Fisher value) means that parameter plays a bigger role in keeping the loss low for the original task.

3. Using It in Fine-tuning:
Once you know which parameters are important (have high Fisher values), you add an extra term to the loss during fine‑tuning that penalizes changes in these parameters. This “reminds” the model of its prior knowledge and helps prevent catastrophic forgetting.

## Pytorch example
The idea is to estimate the importance (Fisher Information) of each parameter on a given task, then add a penalty to the loss during fine‑tuning that discourages significant changes to those important weights.

1. Compute the Fisher Information
You first compute the Fisher Information for each parameter by running your model over data from the previous task and accumulating the squared gradients.

```python
import torch
import torch.nn as nn

class EWC:
    def __init__(self, model: nn.Module, dataloader, device='cpu'):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        # Save a copy of the current parameters
        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        # Compute Fisher Information Matrix
        self.fisher = self._compute_fisher()

    def _compute_fisher(self):
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()
        for data, target in self.dataloader:
            self.model.zero_grad()
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            # Here, we assume your model has a defined loss function (loss_fn)
            loss = self.model.loss_fn(output, target)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
        # Average the fisher information across the dataset
        for n in fisher:
            fisher[n] /= len(self.dataloader)
        return fisher

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                # The penalty term is the weighted squared difference from the original parameters
                loss += (self.fisher[n] * (p - self.params[n]) ** 2).sum()
        return loss
```
2. Integrate the EWC Penalty into Training
During fine‑tuning, add the EWC penalty to your primary loss. You can weight the penalty term with a hyperparameter (e.g., lambda_ewc) to control its influence.

python
복사
## Example training loop incorporating EWC
```python
lambda_ewc = 0.1  # Adjust based on your needs

for data, target in dataloader:
    optimizer.zero_grad()
    data, target = data.to(device), target.to(device)
    output = model(data)
    loss = criterion(output, target)
    
    # EWC penalty from previously computed Fisher information
    ewc_loss = ewc.penalty(model)
    total_loss = loss + lambda_ewc * ewc_loss
    
    total_loss.backward()
    optimizer.step()
    
```
Explanation
Parameter Backup:
We store a copy of the original model parameters (before fine‑tuning) so we can later penalize any significant deviations.

Fisher Information:
The Fisher information is approximated by the squared gradients over a dataset. It indicates the importance of each parameter for the original task.

Penalty Term:
The penalty is computed as the sum of the elementwise squared differences between the current parameters and the saved parameters, weighted by the Fisher information. This term is added to your primary loss to discourage changes to parameters that were important for the previous task.

This approach, originally proposed by Kirkpatrick et al. (2017), helps mitigate catastrophic forgetting by “remembering” the key aspects of the original task while allowing the model to learn new information.

