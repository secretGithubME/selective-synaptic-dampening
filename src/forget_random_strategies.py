"""
Refer to forget_full_class_... for comments
This file is near identical with minimal modifications to facilitate random forgetting.
Seperate file to allow for easy reuse.
"""

import random
import numpy as np
from typing import Tuple, List
from copy import deepcopy

import torch
from torch.utils.data import DataLoader, ConcatDataset, dataset
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F

from sklearn import linear_model, model_selection

from unlearn import *
from metrics import UnLearningScore, get_membership_attack_prob
from utils import *
import ssd as ssd
import conf


def get_metric_scores(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    device,
):
    # Overall test accuracy
    loss_acc_dict = evaluate(model, valid_dl, device)  

    # Retain set accuracy
    retain_acc_dict = evaluate(model, retain_valid_dl, device)  

    # Forget set accuracy
    forget_acc_dict = evaluate(model, forget_valid_dl, device)

    # Zero Retention Force (ZRF) score
    zrf = UnLearningScore(model, unlearning_teacher, forget_valid_dl, 128, device)

    # Membership Inference Attack (MIA) vulnerability
    mia = get_membership_attack_prob(retain_train_dl, forget_train_dl, valid_dl, model)

    # Return all metrics including forget accuracy
    return (
        loss_acc_dict["Acc"],       # Test Accuracy
        retain_acc_dict["Acc"],     # Retain Accuracy
        forget_acc_dict["Acc"],     # Forget Accuracy (newly added)
        zrf,                        # Unlearning Score
        mia                         # Membership Attack Score
    )



def baseline(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    device,
    **kwargs,
):
    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )


def retrain(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    dataset_name,
    model_name,
    device,
    **kwargs,
):
    for layer in model.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
    if model_name == "ViT":
        epochs = getattr(conf, f"{dataset_name}_{model_name}_EPOCHS")
        milestones = getattr(conf, f"{dataset_name}_{model_name}_MILESTONES")
    else:
        epochs = getattr(conf, f"{dataset_name}_EPOCHS")
        milestones = getattr(conf, f"{dataset_name}_MILESTONES")
    _ = fit_one_cycle(
        epochs,
        model,
        retain_train_dl,
        retain_valid_dl,
        milestones=milestones,
        device=device,
    )

    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )


def finetune(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    device,
    **kwargs,
):
    _ = fit_one_cycle(
        5, model, retain_train_dl, retain_valid_dl, lr=0.02, device=device
    )

    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )


def blindspot(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    device,
    **kwargs,
):
    student_model = deepcopy(model)
    KL_temperature = 1
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.0001)
    retain_train_subset = random.sample(
        retain_train_dl.dataset, int(0.3 * len(retain_train_dl.dataset))
    )

    if kwargs["model_name"] == "ViT":
        b_s = 128  # lowered batch size from 256 (original) to fit into memory
    else:
        b_s = 256

    blindspot_unlearner(
        model=student_model,
        unlearning_teacher=unlearning_teacher,
        full_trained_teacher=model,
        retain_data=retain_train_subset,
        forget_data=forget_train_dl.dataset,
        epochs=1,
        optimizer=optimizer,
        lr=0.0001,
        batch_size=b_s,
        device=device,
        KL_temperature=KL_temperature,
    )

    return get_metric_scores(
        student_model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )


def amnesiac(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    num_classes,
    device,
    **kwargs,
):
    unlearninglabels = list(range(num_classes))
    unlearning_trainset = []

    for x, _, clabel in forget_train_dl.dataset:
        rnd = random.choice(unlearninglabels)
        while rnd == clabel:
            rnd = random.choice(unlearninglabels)
        unlearning_trainset.append((x, _, rnd))

    for x, _, y in retain_train_dl.dataset:
        unlearning_trainset.append((x, _, y))

    unlearning_train_set_dl = DataLoader(
        unlearning_trainset, 128, pin_memory=True, shuffle=True
    )

    _ = fit_one_unlearning_cycle(
        3, model, unlearning_train_set_dl, retain_valid_dl, device=device, lr=0.0001
    )
    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )


def FisherForgetting(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    num_classes,
    device,
    **kwargs,
):
    def hessian(dataset, model):
        model.eval()
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        loss_fn = nn.CrossEntropyLoss()

        for p in model.parameters():
            p.grad_acc = 0
            p.grad2_acc = 0

        for data, _, orig_target in tqdm(train_loader):
            data, orig_target = data.to(device), orig_target.to(device)
            output = model(data)
            prob = F.softmax(output, dim=-1).data

            for y in range(output.shape[1]):
                target = torch.empty_like(orig_target).fill_(y)
                loss = loss_fn(output, target)
                model.zero_grad()
                loss.backward(retain_graph=True)
                for p in model.parameters():
                    if p.requires_grad:
                        p.grad_acc += (orig_target == target).float() * p.grad.data
                        p.grad2_acc += prob[:, y] * p.grad.data.pow(2)

        for p in model.parameters():
            p.grad_acc /= len(train_loader)
            p.grad2_acc /= len(train_loader)

    def get_mean_var(p, is_base_dist=False, alpha=3e-6):
        var = deepcopy(1.0 / (p.grad2_acc + 1e-8))
        var = var.clamp(max=1e3)
        if p.size(0) == num_classes:
            var = var.clamp(max=1e2)
        var = alpha * var

        if p.ndim > 1:
            var = var.mean(dim=1, keepdim=True).expand_as(p).clone()
        if not is_base_dist:
            mu = deepcopy(p.data0.clone())
        else:
            mu = deepcopy(p.data0.clone())
        if p.ndim == 1:
            # BatchNorm
            var *= 10
        #         var*=1
        return mu, var

    for p in model.parameters():
        p.data0 = deepcopy(p.data.clone())

    hessian(retain_train_dl.dataset, model)

    fisher_dir = []
    alpha = 1e-6
    for i, p in enumerate(model.parameters()):
        mu, var = get_mean_var(p, False, alpha=alpha)
        p.data = mu + var.sqrt() * torch.empty_like(p.data0).normal_()
        fisher_dir.append(var.sqrt().view(-1).cpu().detach().numpy())
    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )


def ssd_tuning(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    dampening_constant,
    selection_weighting,
    full_train_dl,
    device,
    **kwargs,
):
    parameters = {
        "lower_bound": 1,
        "exponent": 1,
        "magnitude_diff": None,
        "min_layer": -1,
        "max_layer": -1,
        "forget_threshold": 1,
        "dampening_constant": dampening_constant,
        "selection_weighting": selection_weighting,
    }

    # load the trained model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    ssd_t = ssd.ParameterPerturber(model, optimizer, device, parameters)
    model = model.eval()

    sample_importances = ssd_t.calc_importance(forget_train_dl)

    original_importances = ssd_t.calc_importance(full_train_dl)
    ssd_t.modify_weight(original_importances, sample_importances)
    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )

def mu_mis_unlearning(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    dampening_constant,
    selection_weighting,
    full_train_dl,
    device,
    **kwargs,
):
    step_size=0.01 # η: Step length
    stop_threshold=0.1  # δ: Stopping threshold
    max_iterations=10  # TMU: Max iterations
    num_classes = kwargs.get('num_classes')
    model.train()
    model = model.to(device)
    
    # Initialize model parameters
    wt = model.state_dict()  # w0 = wp
    
    # Compute initial loss on forgetting set D_f
    criterion = nn.CrossEntropyLoss()
    initial_loss = 0.0
    for x,_, y in forget_train_dl:
        x, y = x.to(device), y.to(device)
        preds = model(x)
        initial_loss += criterion(preds, y).item()
    L_0 = initial_loss / len(forget_train_dl)  # L̃_0
    
    for t in range(max_iterations):  # Repeat until stopping condition
        print("Epoch: ",t+1)
        delta_w = {key: torch.zeros_like(param) for key, param in wt.items()}  # Initialize Δw
        avg_loss = 0.0  # Initialize L̃
        
        for x,_,y in forget_train_dl:
            x, y = x.to(device), y.to(device)
            
            # Select a random irrelevant class (c' ≠ c)
            random_labels = torch.randint(0, num_classes, y.shape, device=device)
            while torch.any(random_labels == y):  # Ensure c' ≠ c
                random_labels = torch.randint(0, num_classes, y.shape, device=device)
            
            # Compute loss for x with random class
            preds = model(x)
            loss = criterion(preds, random_labels)
            loss.backward()  # Compute gradient
            
            # Accumulate gradient updates
            for key, param in model.named_parameters():
                delta_w[key] += param.grad.clone()
            
            avg_loss += loss.item()
        
        # Normalize updates
        for key in delta_w.keys():
            delta_w[key] = delta_w[key].float() / len(forget_train_dl)
        
        # Update model parameters: wt+1 = wt - ηΔw
        with torch.no_grad():
            for key, param in model.named_parameters():
                param -= step_size * delta_w[key]
        
        # Compute loss change: ΔL̃
        final_loss = 0.0
        for x,_,y in forget_train_dl:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            final_loss += criterion(preds, y).item()
        L_final = final_loss / len(forget_train_dl)
        delta_L = abs(L_0 - L_final) / len(forget_train_dl)
        print("Final loss: ", final_loss)
        
        # Stop if ΔL̃ ≥ δ
        if delta_L >= stop_threshold:
            break
    
    return get_metric_scores(model, unlearning_teacher, retain_train_dl, retain_valid_dl, forget_train_dl, forget_valid_dl, valid_dl, device)

def mu_mis_ssd_tuning(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    dampening_constant,
    selection_weighting,
    full_train_dl,
    device,
    **kwargs,
):
    max_epochs=20
    parameters = {
        "lower_bound": 1,
        "exponent": 1,
        "magnitude_diff": None,
        "min_layer": -1,
        "max_layer": -1,
        "forget_threshold": 1,
        "dampening_constant": dampening_constant,
        "selection_weighting": selection_weighting,
    }

    print("🔹 Improved MU MIS SSD Tuning")

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)

    ssd_t = ssd.ParameterPerturber(model, optimizer, device, parameters)
    importance_forget = ssd_t.calc_importance(forget_train_dl)
    importance_full = ssd_t.calc_importance(full_train_dl)

    w_p = deepcopy(model.state_dict())
    prev_loss = float("inf")
    delta_threshold = 1e-3  # Stopping criteria
    epoch = 0

    while epoch < max_epochs:
        print(f"\n🔄 Epoch {epoch + 1} ----------------------")
        delta_w = {key: torch.zeros_like(param) for key, param in model.state_dict().items()}
        fisher_info = {key: torch.zeros_like(param) for key, param in model.state_dict().items()}

        model.train()
        for batch in forget_train_dl:
            inputs, _, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()

            for key, param in model.named_parameters():
                hessian_term = 0.5 * torch.norm(param.grad) ** 2  # Hessian Approx
                fisher_info[key] += torch.norm(param.grad) ** 2  # Fisher Information
                delta_w[key] += importance_forget[key] * (param.grad + hessian_term)

        print("✅ Gradient updates computed.")

        with torch.no_grad():
            for key, param in model.named_parameters():
                fisher_scaling = torch.exp(-fisher_info[key])
                delta_w[key] *= fisher_scaling  # Adaptive Forgetting Scaling
                param -= 0.1 * delta_w[key]  # Learning Rate applied
                param.copy_(torch.clamp(param, -0.5, 0.5))  # Trust Region Constraint

        print("🔧 Model weights updated.")
        model.eval()

        total_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch in valid_dl:
                images, _, labels = batch
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                total_loss += loss.item()
                num_batches += 1

        new_loss = total_loss / num_batches
        print(f"📉 Validation Loss: {new_loss:.6f} (Previous: {prev_loss:.6f})")

        if abs(prev_loss - new_loss) < delta_threshold:
            print("✅ Convergence reached. Stopping...")
            break

        prev_loss = new_loss
        scheduler.step()
        epoch += 1

    print("🏁 Training complete.")
    return get_metric_scores(model, unlearning_teacher, retain_train_dl, retain_valid_dl, forget_train_dl, forget_valid_dl, valid_dl, device)


import torch
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy

def cluade(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    dampening_constant,
    selection_weighting,
    full_train_dl,
    device,
    **kwargs,
):
    """
    Combined Machine Unlearning Method:
    Integrates Selective Synaptic Dampening (SSD) and Suppressing Sample Contribution (SSC)
    
    Key Improvements:
    - More robust importance calculation
    - Adaptive learning rate
    - Enhanced regularization
    - Improved convergence tracking
    """
    # Hyperparameters
    max_epochs = kwargs.get('max_epochs', 30)
    learning_rate = kwargs.get('learning_rate', 0.01)
    regularization_strength = kwargs.get('regularization_strength', 1e-4)
    
    # Detailed parameters for SSD
    parameters = {
        "lower_bound": kwargs.get('lower_bound', 1),
        "exponent": kwargs.get('exponent', 1),
        "magnitude_diff": kwargs.get('magnitude_diff', None),
        "min_layer": kwargs.get('min_layer', -1),
        "max_layer": kwargs.get('max_layer', -1),
        "forget_threshold": kwargs.get('forget_threshold', 1),
        "dampening_constant": dampening_constant,
        "selection_weighting": selection_weighting,
    }

    print("🔹 Advanced Machine Unlearning: SSD + SSC")

    # Setup optimizer with adaptive learning rate
    optimizer = optim.Adam(model.parameters(), 
                           lr=learning_rate, 
                           weight_decay=regularization_strength)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3, 
        min_lr=1e-5
    )

    # Instantiate SSD Parameter Perturber
    ssd_t = ssd.ParameterPerturber(model, optimizer, device, parameters)
    
    # Calculate importance scores
    importance_forget = ssd_t.calc_importance(forget_train_dl)
    importance_full = ssd_t.calc_importance(full_train_dl)

    # Store initial weights for reference
    initial_weights = deepcopy(model.state_dict())
    
    # Convergence tracking
    best_loss = float('inf')
    patience_counter = 0
    max_patience = 5

    for epoch in range(max_epochs):
        print(f"\n🔄 Epoch {epoch + 1} ----------------------")
        
        # Training phase focusing on forget set
        model.train()
        epoch_loss = 0.0
        total_batches = 0

        for batch in forget_train_dl:
            inputs, _, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # SSC: Suppress sample contribution
            base_loss = F.cross_entropy(outputs, labels)
            
            # SSD: Compute parameter-wise importance loss
            importance_loss = 0.0
            for name, param in model.named_parameters():
                if name in importance_forget:
                    importance_loss += torch.norm(
                        param * importance_forget[name], 
                        p=2
                    )
            
            # Combined loss
            total_loss = base_loss + dampening_constant * importance_loss
            
            # Backward pass
            total_loss.backward()
            
            # Adaptive gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += total_loss.item()
            total_batches += 1

        # Average epoch loss
        avg_loss = epoch_loss / total_batches
        print(f"📉 Epoch Loss: {avg_loss:.6f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in valid_dl:
                images, _, labels = batch
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
                val_loss += loss.item()
                val_batches += 1
        
        val_loss /= val_batches
        print(f"🔍 Validation Loss: {val_loss:.6f}")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping with best model tracking
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        # Convergence and early stopping
        if patience_counter >= max_patience:
            print("🛑 Early stopping triggered")
            model.load_state_dict(best_weights)
            break

    print("🏁 Advanced Unlearning Complete")

    # Final metric evaluation
    return get_metric_scores(
        model, 
        unlearning_teacher, 
        retain_train_dl, 
        retain_valid_dl, 
        forget_train_dl, 
        forget_valid_dl, 
        valid_dl, 
        device
    )
