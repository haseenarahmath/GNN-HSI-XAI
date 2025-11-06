import torch
from captum.attr import IntegratedGradients

def explain_ig(model, x, adj, y, device):
    """
    Returns attributions wrt input features: shape [N, B].
    Captum invocation via a wrapper that fixes adjacency.
    """
    class Wrapper(torch.nn.Module):
        def __init__(self, m, A): super().__init__(); self.m, self.A = m, A
        def forward(self, feats): return self.m(feats, self.A)

    model.eval()
    wrapper = Wrapper(model, adj).to(device)
    x = x.detach().requires_grad_(True)
    ig = IntegratedGradients(wrapper)
    attrs = ig.attribute(
        x, baselines=torch.zeros_like(x, device=device),
        target=y.to(device), n_steps=32, internal_batch_size=1024
    )
    return attrs.abs()
