import torch
from torch import nn

def gradient_attack(logits: torch.Tensor, x: torch.Tensor, y: torch.Tensor,
                    epsilon: float, norm: str = "2",
                    loss_fn=torch.nn.functional.cross_entropy):
    """
    Perform a single-step projected gradient attack (PGD) on the input x.

    Parameters
    ----------
    logits: torch.Tensor of shape [B, K], where B is the batch size and K is 
            the number of classes. 
        The logits for each sample in the batch.
    x: torch.Tensor of shape [B, C, N, N], where B is the batch size, C is the 
       number of channels, and N is the image dimension.
        The input batch of images. Note that x.requires_grad must have been 
        active before computing the logits (otherwise will throw ValueError).
    y: torch.Tensor of shape [B, 1]
        The labels of the input batch of images.
    epsilon: float
        The desired strength of the perturbation. That is, the perturbation 
        (before the projection step) will have a norm of exactly epsilon as 
        measured by the desired norm (see argument: norm). Therefore, epsilon
        implicitly fixes the step size of the PGD update.
    norm: str, can be ["1", "2", "inf"]
        The norm with which to measure the perturbation. E.g., when norm="1", 
        the perturbation (before the projection step) will have a L_1 norm of 
        exactly epsilon (see argument: epsilon).
    loss_fn: function
        The loss function used to construct the attack. By default, this is 
        simply the cross entropy loss.

    Returns
    -------
    torch.Tensor of shape [B, C, N, N]: the perturbed input samples.
    """
    norm = str(norm)
    assert norm in ["1", "2", "inf"]
    ##########################################################
    # YOUR CODE HERE
    x.requires_grad_()
    if x.grad is not None:
        x.grad.zero_()
    
    loss = loss_fn(logits, y)
    loss.backward()

    with torch.no_grad():
        if norm == "1":
            pert = epsilon * x.grad / x.grad.norm(1, dim=(1, 2, 3), keepdim=True)
        elif norm == "2":
            pert = epsilon * x.grad / x.grad.norm(2, dim=(1, 2, 3), keepdim=True)
        elif norm == "inf":
            pert = epsilon * x.grad.sign()
        
        x_pert = x + pert
        x_pert = torch.clamp(x_pert, 0, 1)   
    ##########################################################
    return x_pert.detach()


def attack(x: torch.Tensor, y: torch.Tensor, model: nn.Module, attack_args: dict):
    """
    Run the gradient_attack function above once on x.

    Parameters
    ----------
    x: torch.Tensor of shape [B, C, N, N], where B is the batch size, C is the 
       number of channels, and N is the image dimension.
        The input batch of images. Note that x.requires_grad must have been 
        active before computing the logits (otherwise will throw ValueError).
    y: torch.Tensor of shape [B, 1]
        The labels of the input batch of images.
    model: nn.Module
        The model to be attacked.
    attack_args: dict 
        Additional arguments to be passed to the attack function.

    Returns
    -------
    x_pert: torch.Tensor of the same shape of x 
        Similar as x but perturbed
    y_pert:  torch.Tensor of shape [B, 1]
        Predictions for x_pert
    """

    ##########################################################
    # YOUR CODE HERE
    model.eval()
    x.requires_grad_()
    logits = model(x)
    x_pert = gradient_attack(logits, x, y, **attack_args)
    with torch.no_grad():
        y_pert_logits = model(x_pert)
        y_pert = y_pert_logits.argmax(dim=1)
    ##########################################################
    return x_pert, y_pert
