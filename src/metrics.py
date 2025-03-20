import torch
from lifelines.utils import concordance_index
import torch.nn.functional as F


def evaluate_c_index(data, patient_km_fits, time_col, event_col):
    """
    Evaluates the performance of the nearest-neighbor KM estimator using the C-index.

    Args:
        data: Pandas DataFrame containing patient data.
        patient_km_fits: Dictionary of KaplanMeierFitter objects returned by nearest_neighbor_km.
        time_col: Name of the column representing time-to-event.
        event_col: Name of the column representing the event indicator.

    Returns:
        The C-index.
    """

    true_times = data[time_col].values
    true_events = data[event_col].values
    predicted_median_survival_times = []

    for i in range(len(data)):
        kmf = patient_km_fits[i]
        predicted_median_survival_times.append(kmf.median_survival_time_)

    return concordance_index(true_times, predicted_median_survival_times, true_events)

def mmd(x, y): # linear MMD
    k_xx = torch.mm(x, x.t())
    k_yy = torch.mm(y, y.t())
    k_xy = torch.mm(x, y.t())

    mmd = torch.mean(k_xx) + torch.mean(k_yy) - 2 * torch.mean(k_xy)

    return mmd 

def log_loss(y_pred, Y, eps=1e-8):
    log_loss =  Y * torch.log(y_pred + eps) + (1 - Y) * torch.log(1 - y_pred + eps)
    return -log_loss

def targeted_regularization_loss(y1, y0, t, Y, T, eps, beta=1.0):
    y_pred = y1 * T + y0 * (1 - T)
    q_tilde = y_pred + eps * ((T / t) - (1 - T)/(1 - t))
    gamma = (Y - q_tilde).pow(2)
    targeted_regularization_loss = beta * gamma.mean()
    return targeted_regularization_loss

def dragonnet_loss(y1, y0, t, Y, T, alpha=1.0):
    y_pred = y1 * T + y0 * (1 - T)
    outcome_loss = F.mse_loss(y_pred, Y)
    treatment_loss = F.binary_cross_entropy(t, T)
    dragonnet_loss = outcome_loss + alpha * treatment_loss 
    return dragonnet_loss