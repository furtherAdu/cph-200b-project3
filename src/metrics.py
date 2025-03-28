import torch
import torch.nn.functional as F

def mmd(x, y): # linear MMD
    k_xx = torch.mm(x, x.t())
    k_yy = torch.mm(y, y.t())
    k_xy = torch.mm(x, y.t())

    mmd = torch.mean(k_xx) + torch.mean(k_yy) - 2 * torch.mean(k_xy)

    return mmd 

def log_loss(y_pred, Y, eps=1e-8):
    log_loss =  Y * torch.log(y_pred + eps) + (1 - Y) * torch.log(1 - y_pred + eps)
    return -log_loss

def targeted_regularization_loss(ys, t, Y, T, eps, beta=1.0):
    # y_pred = y1 * T + y0 * (1 - T)
    # q_tilde = y_pred + eps * ((T / t) - (1 - T)/(1 - t))
    T_one_hot = F.one_hot(T.long(), num_classes=len(ys))
    Y_one_hot = F.one_hot(Y.long(), num_classes=len(ys))
    
    y_pred = torch.stack([y * (T==k) for k,y in ys.items()]).argmax(dim=1, keepdim=True)
    
    q_tilde = y_pred + eps * (T_one_hot / t) - (1-T_one_hot)/(1-t)
    gamma = (Y_one_hot- q_tilde).pow(2)
    targeted_regularization_loss = beta * gamma.mean()
    
    print('Targeted regularization loss:', targeted_regularization_loss.item())
    
    return targeted_regularization_loss

def dragonnet_loss(ys, t, Y, T, alpha=1.0):
    # y_pred = y1 * T + y0 * (1 - T)
    outcome_loss = mse_loss(ys, Y, T)
    
    if len(ys) == 2:
        treatment_loss = F.binary_cross_entropy(t, T.long())
    else:
        T_one_hot = F.one_hot(T.long(), num_classes=len(ys))
        treatment_loss = F.cross_entropy(t, T_one_hot)
        
    dragonnet_loss = outcome_loss + alpha * treatment_loss 
    return dragonnet_loss

def mse_loss(self, ys, Y, T):
    
    y_pred = torch.stack([y * (T==k) for k,y in ys.items()]).sum(1)

    if self.continuous_outcome:
        mse_loss = F.mse_loss(y_pred, Y)
    else:
        Y_one_hot = F.one_hot(Y.long(), num_classes=len(ys))
        mse_loss = F.mse_loss(y_pred, Y_one_hot)
        
    return mse_loss


def balanced_accuracy(probs, labels):
    """
    Calculates balanced accuracy from predicted class probabilities and true labels.

    Args:
        probs (torch.Tensor): Predicted class probabilities (n x k).
        labels (torch.Tensor): True class labels (n).

    Returns:
        float: Balanced accuracy.
    """

    preds = torch.argmax(probs, dim=1)  # Get predicted class labels
    unique_labels = torch.unique(labels)
    recalls = []

    for label in unique_labels:
        mask = (labels == label)
        true_positives = torch.sum(preds[mask] == labels[mask]).float()
        total_positives = torch.sum(mask).float()
        recall = true_positives / total_positives
        recalls.append(recall)

    return torch.mean(torch.tensor(recalls)).item()



