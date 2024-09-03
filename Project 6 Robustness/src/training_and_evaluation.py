from typing import Callable, Union, Tuple, List, Dict
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import cross_entropy
import logging
from .attacks import gradient_attack


def train_model(model: nn.Module, dataset: Dataset, batch_size: int,
                device: Union[torch.device, str],
                optimizer: Optimizer, epochs: int = 1,
                loss_function: Callable = None,
                loss_args: Union[dict, None] = None) -> Tuple[List, List]:
    """
    Train a model on the input dataset.

    Parameters
    ----------
    model: nn.Module
        The input model to be trained.
    dataset: torch.utils.data.Dataset
        The dataset to train on.
    batch_size: int
        The training batch size.
    device: torch.device or str
        The calculation device.
    optimizer: Optimizer
        The model's optimizer.
    epochs: int
        Number of epochs to train for. Default: 1.
    loss_args: dict or None
        Additional arguments to be passed to the loss function.

    Returns
    -------
    Tuple containing
        * losses: List[float]. The losses obtained at each step.
        * accuracies: List[float]. The accuracies obtained at each step.
    """
    if loss_args is None:
        loss_args = {}
    losses = []
    accuracies = []
    num_train_batches = torch.tensor(len(dataset) / batch_size)
    num_train_batches = int(torch.ceil(num_train_batches).item())
    for epoch in range(epochs):
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for i, (x, y) in enumerate(train_loader):
            loss, accuracy = training_step(x, y, model, loss_function,
                                           loss_args, optimizer, device)
            losses.append(loss.detach())
            accuracies.append(accuracy)
            if i % 100 == 0:
                logging.info(
                    f"Epoch {epoch} Iteration {i}: Loss={loss} Accuracy={accuracy}")
    return losses, accuracies


def training_step(x: torch.Tensor, y: torch.Tensor, model: nn.Module,
                  loss_function: Callable, loss_args: Union[dict, None],
                  optimizer: Optimizer, device: Union[torch.device, str]):
    """
        Performs a single training step.

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
    optimizer: Optimizer
        The model's optimizer.
    device: torch.device or str
        The calculation device.

    Returns
    ----------
    loss: torch.Tensor of shape [1]
        The loss of a single training step (for the batch)
    accuracy: torch.Tensor of shape [1]
        The accuracy of a single training step (for the batch)
    """
    ##########################################################
    # YOUR CODE HERE
    model.train()
    x = x.to(device)
    y = y.to(device)

    x.requires_grad_()
    loss, logits = standard_loss(model, x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    preds = logits.argmax(dim=1)
    accuracy = (preds == y).float().mean()
    ##########################################################
    return loss, accuracy


def standard_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor):
    '''
        Computes standard cross-entropy loss.
        You can use the imported function.

    Parameters
    ----------
    x: torch.Tensor of shape [B, C, N, N], where B is the batch size, C is the 
       number of channels, and N is the image dimension.
       The input batch of images. Note that x.requires_grad must have been 
       active before computing the logits (otherwise will throw ValueError).
    y: torch.Tensor of shape [B, 1]
        The labels of the input batch of images.
    model: nn.Module
        The input model to be used.

    Returns
    -------
    Tuple containing
        * loss: torch.Tensor, scalar
            Mean cross-entropy loss 
        * logits: torch.Tensor, shape [B, K], K is the number of classes
            The obtained logits 
    '''
    ##########################################################
    # YOUR CODE HERE
    x.requires_grad_()
    logits = model(x)
    loss = cross_entropy(logits, y)
    loss.item()
    ##########################################################
    return loss, logits


def predict_model(model: nn.Module, dataset: Dataset, batch_size: int,
                  device: Union[torch.device, str],
                  attack_function: Union[Callable, None] = None,
                  attack_args: Union[Callable, None] = None) -> float:
    """
    Use the model to predict a label for each sample in the provided dataset. 

    Optionally performs an attack via the attack function first.

    Parameters
    ----------
    model: nn.Module
        The input model to be used.
    dataset: torch.utils.data.Dataset
        The dataset to predict for.
    batch_size: int
        The batch size.
    device: torch.device or str
        The calculation device.
    attack_function: function or None
        If not None, call the function to obtain a perturbed batch before 
        evaluating the prediction.
    attack_args: dict or None
        Additional arguments to be passed to the attack function.

    Returns
    -------
    float: the accuracy on the provided dataset.
    """
    model.eval()
    if attack_args is None:
        attack_args = {}
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    num_batches = torch.tensor(len(dataset) / batch_size)
    num_batches = int(torch.ceil(num_batches).item())
    predictions = []
    targets = []
    for x, y in test_loader:
        targets.append(y)
        logits = prediction(x, y, model, attack_function, attack_args, device)
        predictions.append(logits.argmax(-1).cpu())
    predictions = torch.cat(predictions)
    targets = torch.cat(targets)
    accuracy = (predictions == targets).float().mean().item()
    return accuracy


def prediction(x: torch.Tensor, y: torch.Tensor, model: nn.Module,
               attack_function: Union[Callable, None],
               attack_args: Union[Callable, None],
               device: Union[torch.device, str]):
    '''
        Use the model to predict a label for a given sample (x,y).

        Optionally performs an attack via the attack function first.

    Parameters
    ----------
    x: torch.Tensor of shape [B, C, N, N], where B is the batch size, C is the 
       number of channels, and N is the image dimension.
        The input batch of images. Note that x.requires_grad must have been 
        active before computing the logits (otherwise will throw ValueError).
    y: torch.Tensor of shape [B, 1]
        The labels of the input batch of images.
    model: nn.Module
        The input model to be used.
    attack_function: function or None
        If not None, call the function to obtain a perturbed batch before 
        evaluating the prediction.
    attack_args: dict or None
        Additional arguments to be passed to the attack function.
    device: torch.device or str
        The calculation device.

    Returns
    -------
    logits: torch.Tensor of shape [B,K]
    '''
    ##########################################################
    # YOUR CODE HERE
    if attack_function is not None:
        x = x.to(device)
        x.requires_grad_()
        x_log = model(x)
        epsilon = attack_args['epsilon']
        norm = attack_args['norm']
        x = attack_function(x_log, x, y, epsilon, norm)
    else:
        x = x.to(device)
        x.requires_grad_()
   
    logits = model(x)
    ##########################################################
    return logits


def loss_function_adversarial_training(model: nn.Module, x: torch.Tensor, y: torch.Tensor, **attack_args):
    """
    Loss function used for adversarial training. First computes adversarial 
    examples on the input batch via gradient_attack and then computes the 
    logits and the loss on the adversarial examples.
    Parameters
    ----------
    x: torch.Tensor of shape [B, C, N, N], where B is the batch size, C is the 
       number of channels, and N is the image width/height.
        The input batch to certify.
    y: torch.Tensor of shape [B, 1].
        The labels of the input batch.
    model: torch.nn.Module
        The classifier to be evaluated.
    attack_args: 
        additional arguments passed to the adversarial attack function.

    Returns
    -------
    Tuple containing
        * loss_pert: torch.Tensor, scalar
            Mean loss obtained on the adversarial examples.
        * logits_pert: torch.Tensor, shape [B, K], K is the number of classes
            The logits obtained on the adversarial examples.
    """
    ##########################################################
    # YOUR CODE HERE
    x.requires_grad_()
    logits = model(x)
    adv_ex = gradient_attack(logits, x, y, **attack_args)

    loss_pert, logits_pert = standard_loss(model, adv_ex, y)
    ##########################################################
    return loss_pert, logits_pert


def evaluate_robustness_smoothing(smoothed_classifier: nn.Module,
                                  sigma: float,
                                  dataset: Dataset,
                                  device: Union[torch.device, str],
                                  num_samples_1: int = 1000,
                                  num_samples_2: int = 10000,
                                  alpha: float = 0.05,
                                  certification_batch_size: float = 5000,
                                  num_classes: int = 10) -> Dict:
    """
    Evaluate the robustness of a smooth classifier based on the input base 
    classifier via randomized smoothing.

    Returns the radius averaged over all predictions (we define the radius as zero for false or abstained predictions).

    Parameters
    ----------
    smoothed_classifier: nn.Module
        The input base classifier to use in the randomized smoothing process.
    sigma: float
        The variance to use for the Gaussian noise samples.
    dataset: Dataset
        The input dataset to predict on.
    device: torch.device or str
        The calculation device.
    num_samples_1: int
        The number of samples used to determine the most likely class.
    num_samples_2: int
        The number of samples used to perform the certification.
    alpha: float
        The desired confidence level that the top class is indeed the most 
        likely class. E.g. alpha=0.05 means that the expected error rate must 
        not be larger than 5%.
    certification_batch_size: int
        The batch size to use during the certification, i.e. how many noise 
        samples to classify in parallel.
    num_classes: int
        The number of classes.

    Returns
    -------
    Dict containing the following keys:
        * abstains: int. The number of times the smooth classifier abstained, 
            i.e. could not certify the input sample to the desired 
            confidence level.
        * false_predictions: int. The number of times the prediction could be 
            certified but was not correct.
        * correct_certified: int. The number of times the prediction could be 
            certified and was correct.
        * avg_radius: float. The average radius for which the predictions could 
            be certified.
    """
    smoothed_classifier.eval()
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    abstains = 0
    false_predictions = 0
    correct_certified = 0
    radii = []
    for x, y in test_loader:
        ##########################################################
        # YOUR CODE HERE
        x = x.to(device)
        y = y.to(device)
        class_pred = smoothed_classifier.predict(x, num_samples_1, alpha, certification_batch_size)
        if class_pred == -1:
            abstains +=1
            radii.append(0.0)
        else:
            class_cert, radius = smoothed_classifier.certify(x, num_samples_1, num_samples_2, alpha, certification_batch_size)
            if class_cert == -1:
                abstains += 1
                radii.append(0.0)
            elif class_cert == y.item():
                correct_certified += 1
                radii.append(radius)
            else:
                false_predictions += 1
                radii.append(0.0)
        ##########################################################

    avg_radius = torch.tensor(radii).mean().item()
    return dict(abstains=abstains, false_predictions=false_predictions,
                correct_certified=correct_certified, avg_radius=avg_radius)
