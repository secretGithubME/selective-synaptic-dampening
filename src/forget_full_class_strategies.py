import random
import numpy as np
from typing import Tuple, List
from copy import deepcopy

import torch
from torch.utils.data import DataLoader, ConcatDataset, dataset
import itertools

from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from sklearn import linear_model, model_selection
from collections import OrderedDict
import torch.nn as nn


from unlearn import *
from metrics import UnLearningScore, get_membership_attack_prob
from utils import *
import ssd as ssd
import conf
import timeit


# Create datasets of the classes
def get_classwise_ds(ds, num_classes):
    classwise_ds = {}
    for i in range(num_classes):
        classwise_ds[i] = []

    for img, label, clabel in ds:
        classwise_ds[clabel].append((img, label, clabel))
    return classwise_ds


# Creates datasets for method execution
def build_retain_forget_sets(
    classwise_train, classwise_test, num_classes, forget_class
):
    # Getting the forget and retain validation data
    forget_valid = []
    for cls in range(num_classes):
        if cls == forget_class:
            for img, label, clabel in classwise_test[cls]:
                forget_valid.append((img, label, clabel))

    retain_valid = []
    for cls in range(num_classes):
        if cls != forget_class:
            for img, label, clabel in classwise_test[cls]:
                retain_valid.append((img, label, clabel))

    forget_train = []
    for cls in range(num_classes):
        if cls == forget_class:
            for img, label, clabel in classwise_train[cls]:
                forget_train.append((img, label, clabel))

    retain_train = []
    for cls in range(num_classes):
        if cls != forget_class:
            for img, label, clabel in classwise_train[cls]:
                retain_train.append((img, label, clabel))

    return (retain_train, retain_valid, forget_train, forget_valid)


# Returns metrics
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
    loss_acc_dict = evaluate(model, valid_dl, device)
    retain_acc_dict = evaluate(model, retain_valid_dl, device)
    zrf = UnLearningScore(model, unlearning_teacher, forget_valid_dl, 128, device)
    d_f = evaluate(model, forget_valid_dl, device)
    mia = get_membership_attack_prob(retain_train_dl, forget_train_dl, valid_dl, model)

    return (loss_acc_dict["Acc"], retain_acc_dict["Acc"], zrf, mia, d_f["Acc"])


# Does nothing; original model
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


# Retrain the model on the retrain dataset only
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


# Finetune the model using the retain data for a set number of epochs
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


# Bad Teacher from https://github.com/vikram2000b/bad-teaching-unlearning
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


# Implementation from https://github.com/vikram2000b/bad-teaching-unlearning
def amnesiac(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    num_classes,
    forget_class,
    device,
    **kwargs,
):
    unlearninglabels = list(range(num_classes))
    unlearning_trainset = []

    unlearninglabels.remove(forget_class)

    for x, _, clabel in forget_train_dl.dataset:
        unlearning_trainset.append((x, _, random.choice(unlearninglabels)))

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


# Extremely slow >>> Fisher https://github.com/AdityaGolatkar/SelectiveForgetting
def NTK(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    forget_class,
    num_classes,
    device,
    **kwargs,
):
    def delta_w_utils(model_init, dataloader, name="complete"):
        model_init.eval()
        dataloader = torch.utils.data.DataLoader(
            dataloader.dataset, batch_size=1, shuffle=False
        )
        G_list = []
        f0_minus_y = []
        for idx, batch in enumerate(
            tqdm(dataloader)
        ):  # (tqdm(dataloader,leave=False)):
            batch = [
                tensor.to(next(model_init.parameters()).device) for tensor in batch
            ]
            input, _, target = batch

            target = target.cpu().detach().numpy()
            output = model_init(input)
            G_sample = []
            for cls in range(num_classes):
                grads = torch.autograd.grad(
                    output[0, cls], model_init.parameters(), retain_graph=True
                )
                grads = np.concatenate([g.view(-1).cpu().numpy() for g in grads])
                G_sample.append(grads)
                G_list.append(grads)
                p = (
                    torch.nn.functional.softmax(output, dim=1)
                    .cpu()
                    .detach()
                    .numpy()
                    .transpose()
                )
                p[target] -= 1
                f0_y_update = deepcopy(p)
            f0_minus_y.append(f0_y_update)
        return np.stack(G_list).transpose(), np.vstack(f0_minus_y)

    #############################################################################################
    model_init = deepcopy(model)
    G_r, f0_minus_y_r = delta_w_utils(deepcopy(model), retain_train_dl, "complete")
    print("GOT GR")
    # np.save('NTK_data/G_r.npy',G_r)
    # np.save('NTK_data/f0_minus_y_r.npy',f0_minus_y_r)
    # del G_r, f0_minus_y_r

    G_f, f0_minus_y_f = delta_w_utils(deepcopy(model), forget_train_dl, "retain")
    print("GOT GF")
    # np.save('NTK_data/G_f.npy',G_f)
    # np.save('NTK_data/f0_minus_y_f.npy',f0_minus_y_f)
    # del G_f, f0_minus_y_f

    # G_r = np.load('NTK_data/G_r.npy')
    # G_f = np.load('NTK_data/G_f.npy')
    G = np.concatenate([G_r, G_f], axis=1)
    print("GOT G")
    # np.save('NTK_data/G.npy',G)
    # del G, G_f, G_r

    # f0_minus_y_r = np.load('NTK_data/f0_minus_y_r.npy')
    # f0_minus_y_f = np.load('NTK_data/f0_minus_y_f.npy')
    f0_minus_y = np.concatenate([f0_minus_y_r, f0_minus_y_f])

    # np.save('NTK_data/f0_minus_y.npy',f0_minus_y)
    # del f0_minus_y, f0_minus_y_r, f0_minus_y_f

    weight_decay = 0.1

    # G = np.load('NTK_data/G.npy')
    theta = G.transpose().dot(G) + (
        len(retain_train_dl.dataset) + len(forget_train_dl.dataset)
    ) * weight_decay * np.eye(G.shape[1])
    # del G

    theta_inv = np.linalg.inv(theta)

    # np.save('NTK_data/theta.npy',theta)
    # del theta

    # G = np.load('NTK_data/G.npy')
    # f0_minus_y = np.load('NTK_data/f0_minus_y.npy')
    w_complete = -G.dot(theta_inv.dot(f0_minus_y))

    # np.save('NTK_data/theta_inv.npy',theta_inv)
    # np.save('NTK_data/w_complete.npy',w_complete)
    # del G, f0_minus_y, theta_inv, w_complete

    # G_r = np.load('NTK_data/G_r.npy')
    num_to_retain = len(retain_train_dl.dataset)
    theta_r = G_r.transpose().dot(G_r) + num_to_retain * weight_decay * np.eye(
        G_r.shape[1]
    )
    # del G_r

    theta_r_inv = np.linalg.inv(theta_r)
    # np.save('NTK_data/theta_r.npy',theta_r)
    # del theta_r

    # G_r = np.load('NTK_data/G_r.npy')
    # f0_minus_y_r = np.load('NTK_data/f0_minus_y_r.npy')
    w_retain = -G_r.dot(theta_r_inv.dot(f0_minus_y_r))

    # np.save('NTK_data/theta_r_inv.npy',theta_r_inv)
    # np.save('NTK_data/w_retain.npy',w_retain)
    # del G_r, f0_minus_y_r, theta_r_inv, w_retain

    def get_delta_w_dict(delta_w, model):
        # Give normalized delta_w
        delta_w_dict = OrderedDict()
        params_visited = 0
        for k, p in model.named_parameters():
            num_params = np.prod(list(p.shape))
            update_params = delta_w[params_visited : params_visited + num_params]
            delta_w_dict[k] = torch.Tensor(update_params).view_as(p)
            params_visited += num_params
        return delta_w_dict

    #### Scrubbing Direction
    # w_complete = np.load('NTK_data/w_complete.npy')
    # w_retain = np.load('NTK_data/w_retain.npy')
    print("got prelims, calculating delta_w")
    delta_w = (w_retain - w_complete).squeeze()
    print("got delta_w")
    # delta_w_copy = deepcopy(delta_w)
    # delta_w_actual = vectorize_params(model0)-vectorize_params(model)

    # print(f'Actual Norm-: {np.linalg.norm(delta_w_actual)}')
    # print(f'Predtn Norm-: {np.linalg.norm(delta_w)}')
    # scale_ratio = np.linalg.norm(delta_w_actual)/np.linalg.norm(delta_w)
    # print('Actual Scale: {}'.format(scale_ratio))
    # log_dict['actual_scale_ratio']=scale_ratio
    def vectorize_params(model):
        param = []
        for p in model.parameters():
            param.append(p.data.view(-1).cpu().numpy())
        return np.concatenate(param)

    m_pred_error = (
        vectorize_params(model) - vectorize_params(model_init) - w_retain.squeeze()
    )
    print(f"Delta w -------: {np.linalg.norm(delta_w)}")

    inner = np.inner(
        delta_w / np.linalg.norm(delta_w), m_pred_error / np.linalg.norm(m_pred_error)
    )
    print(f"Inner Product--: {inner}")

    if inner < 0:
        angle = np.arccos(inner) - np.pi / 2
        print(f"Angle----------:  {angle}")

        predicted_norm = np.linalg.norm(delta_w) + 2 * np.sin(angle) * np.linalg.norm(
            m_pred_error
        )
        print(f"Pred Act Norm--:  {predicted_norm}")
    else:
        angle = np.arccos(inner)
        print(f"Angle----------:  {angle}")

        predicted_norm = np.linalg.norm(delta_w) + 2 * np.cos(angle) * np.linalg.norm(
            m_pred_error
        )
        print(f"Pred Act Norm--:  {predicted_norm}")

    predicted_scale = predicted_norm / np.linalg.norm(delta_w)
    predicted_scale
    print(f"Predicted Scale:  {predicted_scale}")
    # log_dict['predicted_scale_ratio']=predicted_scale

    # def NIP(v1,v2):
    #     nip = (np.inner(v1/np.linalg.norm(v1),v2/np.linalg.norm(v2)))
    #     print(nip)
    #     return nip
    # nip=NIP(delta_w_actual,delta_w)
    # log_dict['nip']=nip
    scale = predicted_scale
    direction = get_delta_w_dict(delta_w, model)

    for k, p in model.named_parameters():
        p.data += (direction[k] * scale).to(device)

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


# Implementation from https://github.com/vikram2000b/Fast-Machine-Unlearning
def UNSIR(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    num_classes,
    forget_class,
    device,
    **kwargs,
):
    classwise_train = get_classwise_ds(
        ConcatDataset((retain_train_dl.dataset, forget_train_dl.dataset)), num_classes
    )
    noise_batch_size = 32
    retain_valid_dl = DataLoader(retain_valid_dl.dataset, batch_size=noise_batch_size)
    # collect some samples from each class
    num_samples = 500
    retain_samples = []
    for i in range(num_classes):
        if i != forget_class:
            retain_samples += classwise_train[i][:num_samples]

    forget_class_label = forget_class
    img_shape = next(iter(retain_train_dl.dataset))[0].shape[-1]
    noise = UNSIR_noise(noise_batch_size, 3, img_shape, img_shape).to(device)
    noise = UNSIR_noise_train(
        noise, model, forget_class_label, 25, noise_batch_size, device=device
    )
    noisy_loader = UNSIR_create_noisy_loader(
        noise,
        forget_class_label,
        retain_samples,
        batch_size=noise_batch_size,
        device=device,
    )
    # impair step
    _ = fit_one_unlearning_cycle(
        1, model, noisy_loader, retain_valid_dl, device=device, lr=0.0001
    )
    # repair step
    other_samples = []
    for i in range(len(retain_samples)):
        other_samples.append(
            (
                retain_samples[i][0].cpu(),
                torch.tensor(retain_samples[i][2]),
                torch.tensor(retain_samples[i][2]),
            )
        )

    heal_loader = torch.utils.data.DataLoader(
        other_samples, batch_size=128, shuffle=True
    )
    _ = fit_one_unlearning_cycle(
        1, model, heal_loader, retain_valid_dl, device=device, lr=0.0001
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


# Ours
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
        "lower_bound": 1,  # unused
        "exponent": 1,  # unused
        "magnitude_diff": None,  # unused
        "min_layer": -1,  # -1: all layers are available for modification
        "max_layer": -1,  # -1: all layers are available for modification
        "forget_threshold": 1,  # unused
        "dampening_constant": dampening_constant,  # Lambda from paper
        "selection_weighting": selection_weighting,  # Alpha from paper
    }

    # load the trained model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    pdr = ssd.ParameterPerturber(model, optimizer, device, parameters)

    model = model.eval()

    # Calculation of the forget set importances
    sample_importances = pdr.calc_importance(forget_train_dl)

    # Calculate the importances of D (see paper); this can also be done at any point before forgetting.
    original_importances = pdr.calc_importance(full_train_dl)

    # Dampen selected parameters
    pdr.modify_weight(original_importances, sample_importances)

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
    step_size=0.05 # η: Step length
    stop_threshold=0.7  # δ: Stopping threshold
    max_iterations=20  # TMU: Max iterations
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

def MESD_unlearning(
    model,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    device,
    **kwargs
):
    """
    Memory Editing with Self-Distillation (MESD) for machine unlearning.
    """
    alpha=0.5   # Knowledge drift factor
    beta=0.1   # Self-distillation weight
    pruning_threshold=1e-6 # Adaptive neuron pruning threshold
    num_epochs = 10
    
    print("\n[STEP 1] Training Student Model via Self-Distillation...\n")
    
    student_model = deepcopy(model)
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.0005)
    
    for epoch in range(num_epochs):  
        total_loss = 0.0
        for x,_, y in retain_train_dl:
            x, y = x.to(device), y.to(device)
            teacher_logits = model(x).detach()
            student_logits = student_model(x)
            
            loss = F.kl_div(F.log_softmax(student_logits, dim=-1), 
                            F.softmax(teacher_logits, dim=-1), 
                            reduction="batchmean") * beta
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs} - Self-Distillation Loss: {total_loss:.4f}")

    print("\n[STEP 2] Injecting Controlled Noise for Knowledge Drift...\n")

    # Store original weights before modification
    original_params = {name: p.clone().detach() for name, p in student_model.named_parameters()}

    with torch.no_grad():
        for p in student_model.parameters():
            noise = alpha * torch.randn_like(p) * p.std()
            p.add_(noise)

    # Measure drift
    total_drift = 0.0
    for name, p in student_model.named_parameters():
        drift = torch.norm(p - original_params[name]).item()
        total_drift += drift
        print(f"Layer {name}: Weight Drift = {drift:.6f}")

    print(f"\nTotal Model Drift: {total_drift:.6f}\n")

    print("\n[STEP 3] Applying Contrastive Forgetting Loss (CFL)...\n")

    def contrastive_forgetting_loss(model, x_forget, x_retain):
        forget_feats = model(x_forget)
        retain_feats = model(x_retain)

        # Ensure both tensors have the same batch size
        min_size = min(forget_feats.shape[0], retain_feats.shape[0])
        forget_feats = forget_feats[:min_size]
        retain_feats = retain_feats[:min_size]

        return -F.cosine_similarity(forget_feats, retain_feats).mean()  # Maximize difference

    forget_batch = next(iter(forget_train_dl))[0].to(device)
    retain_batch = next(iter(retain_train_dl))[0].to(device)
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.0001)

    for epoch in range(2):
        optimizer.zero_grad()
        loss = contrastive_forgetting_loss(student_model, forget_batch, retain_batch)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/2 - Contrastive Forgetting Loss: {loss.item():.6f}")

    print("\n[STEP 4] Pruning Neurons Responsible for Forgotten Classes...\n")

    def prune_model(model, threshold=pruning_threshold):
        prune_count = 0
        with torch.no_grad():
            for p in model.parameters():
                mean_abs = p.abs().mean()
                if mean_abs < threshold:
                    prune_count += 1
                    p.zero_()  # Prune neuron
        
        print(f"Total Neurons Pruned: {prune_count}")

    prune_model(student_model)

    print("\n[STEP 5] Generating Pseudo-Examples for Active Misremembering...\n")

    synthetic_data = torch.randn_like(forget_batch) * 0.1 + forget_batch
    with torch.no_grad():
        student_model(synthetic_data)  # Feed pseudo-examples into model

    print("MESD Unlearning Completed!\n")

    return get_metric_scores(
        student_model,
        model,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    ) 

def SuperUnlearning(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    forget_class,
    num_classes,
    device,
    dataset_name,
    model_name,
    **kwargs,
):
    """
    Implementation of Super Unlearning model that integrates multiple unlearning techniques.
    
    This model strategically combines:
    1. Projection-based weight modification
    2. Low-rank weight updates
    3. Knowledge distillation with selective forgetting
    4. Neuron pruning for critical neurons
    5. Targeted fine-tuning
    
    The combined approach provides stronger forgetting guarantees while maintaining
    performance on retained data.
    """
    print("Starting Super Unlearning...")
    
    # Step 1: Save original model for knowledge distillation
    unlearning_teacher.load_state_dict(model.state_dict())
    unlearning_teacher.eval()
    
    # Step 2: Projection-based forgetting on the final layer
    print("Phase 1: Projection-based weight modification...")
    
    # Get feature extractor (everything before the final layer)
    if hasattr(model, "fc"):  # ResNet, VGG-like models
        final_layer = model.fc
        feature_layers = nn.Sequential(*list(model.children())[:-1])
    elif hasattr(model, "classifier"):  # Some VGG variants, ViT
        final_layer = model.classifier
        feature_layers = nn.Sequential(*list(model.children())[:-1])
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")
    
    # Save original weights of the final layer
    original_weights = final_layer.weight.data.clone()
    
    # Extract features from retain set
    print("Extracting features from retain set...")
    all_features = []
    for batch in tqdm(retain_train_dl, desc="Extracting features"):
        inputs, _, targets = [tensor.to(device) for tensor in batch]
        with torch.no_grad():
            # Get features before the final layer
            if hasattr(model, "fc"):
                features = feature_layers(inputs).squeeze()
            else:
                features = feature_layers(inputs)
                features = torch.flatten(features, 1)
            all_features.append(features)
    
    # Stack all features
    all_features = torch.cat(all_features, dim=0)
    
    # Calculate projection matrix (XᵀX)(XᵀX)⁻¹
    print("Computing projection matrix...")
    X = all_features.cpu()
    XtX = X.T @ X
    # Use pseudoinverse for stability
    XtX_inv = torch.linalg.pinv(XtX)
    projection_matrix = XtX @ XtX_inv
    
    # Project the weights
    print("Projecting weights...")
    projected_weights = (projection_matrix @ original_weights.cpu().T).T
    
    # Update the model with projected weights
    final_layer.weight.data = projected_weights.to(device)
    
    # Step 3: Low-rank weight modification for deeper layers
    print("Phase 2: Low-rank weight updates...")
    
    # Compute gradients for forget and retain sets
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    # Function to compute average gradients
    def compute_gradients(dataloader):
        grads = {name: torch.zeros_like(param.data) for name, param in model.named_parameters()}
        count = 0
        
        for batch in tqdm(dataloader, desc="Computing gradients"):
            inputs, _, targets = [tensor.to(device) for tensor in batch]
            count += inputs.size(0)
            
            model.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grads[name] += param.grad.data * inputs.size(0)  # Weighted by batch size
        
        # Normalize by total number of samples
        for name in grads:
            grads[name] /= count
            
        return grads
    
    # Compute gradients for forget and retain sets
    forget_grads = compute_gradients(forget_train_dl)
    retain_grads = compute_gradients(retain_train_dl)
    
    # Compute the orthogonal direction to remove forget knowledge
    update_direction = {}
    
    for name, param in model.named_parameters():
        if name in forget_grads and param.dim() > 1:  # Only update matrices, not biases
            # Skip the final layer as it's already handled by projection
            if "fc" in name or "classifier" in name:
                continue
                
            # Reshape to 2D for matrix computations
            original_shape = param.data.shape
            F_grad = forget_grads[name].view(original_shape[0], -1)
            R_grad = retain_grads[name].view(original_shape[0], -1)
            
            # Compute SVD of retain gradients to find subspace
            try:
                U, S, V = torch.svd(R_grad)
                
                # Determine rank based on singular values
                threshold = 0.01 * S.max()
                rank = (S > threshold).sum().item()
                rank = max(1, min(rank, min(U.shape[0], V.shape[0]) - 1))  # Ensure valid rank
                
                # Project forget gradient onto orthogonal complement of retain subspace
                U_r = U[:, :rank]  # Retain subspace
                
                # Project forget gradient onto orthogonal complement of U_r
                proj = F_grad - U_r @ (U_r.T @ F_grad)
                
                # Scale the update - negative to move away from forget class
                alpha = 3.0  # Scaling factor for unlearning
                update = -alpha * proj.view(original_shape)
                
                update_direction[name] = update
            except RuntimeError:
                # SVD might fail for small or ill-conditioned matrices - use simpler approach
                update_direction[name] = -alpha * forget_grads[name]
        else:
            update_direction[name] = torch.zeros_like(param.data)
    
    # Apply the update direction
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in update_direction:
                param.data += update_direction[name]
    
    # Step 4: Identify and prune neurons critical for forget class
    print("Phase 3: Critical neuron identification and pruning...")
    
    # Track activations for convolutional and linear layers
    activations = {}
    handles = []
    
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hooks to get activations
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(get_activation(name)))
    
    # Get average activations for forget class samples
    model.eval()
    forget_activations = {name: 0 for name in activations}
    sample_count = 0
    
    for batch in tqdm(forget_train_dl, desc="Computing critical neurons"):
        inputs, _, targets = [tensor.to(device) for tensor in batch]
        _ = model(inputs)
        
        for name in activations:
            # For each layer, accumulate activation values
            if len(activations[name].shape) == 4:  # Conv layer
                channel_mean = activations[name].mean(dim=[0, 2, 3])  # Average over batch, height, width
                try:
                  forget_activations[name] += channel_mean
                except KeyError:
                  continue
            else:  # Linear layer
                neuron_mean = activations[name].mean(dim=0)
                try:  # Average over batch
                  forget_activations[name] += neuron_mean
                except KeyError:
                  continue
                
        sample_count += inputs.size(0)
    
    # Normalize to get average activations
    for name in forget_activations:
        forget_activations[name] /= sample_count
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    # Create masks to prune critical neurons
    masks = {}
    prune_ratio = 0.08  # Prune top 8% of neurons most active for forget class
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and name in forget_activations:
            activations_val = forget_activations[name]
            if isinstance(module, nn.Conv2d):
                # For Conv layers, mask channels
                num_to_prune = max(1, int(activations_val.size(0) * prune_ratio))
                _, indices = torch.topk(activations_val, num_to_prune)
                mask = torch.ones_like(activations_val)
                mask[indices] = 0.1  # Attenuate by 90% rather than zero out completely
                masks[name] = mask
            elif isinstance(module, nn.Linear):
                # For Linear layers, mask neurons
                num_to_prune = max(1, int(activations_val.size(0) * prune_ratio))
                _, indices = torch.topk(activations_val, num_to_prune)
                mask = torch.ones_like(activations_val)
                mask[indices] = 0.1  # Attenuate by 90%
                masks[name] = mask
    
    # Apply masks during forward pass
    class MaskedForward:
        def __init__(self, module, mask):
            self.module = module
            self.mask = mask
            self.original_forward = module.forward
        
        def __call__(self, x):
            output = self.original_forward(x)
            if len(output.shape) == 4:  # Conv output
                return output * self.mask.view(1, -1, 1, 1)
            else:  # Linear output
                return output * self.mask
    
    # Apply masking to model
    for name, module in model.named_modules():
        if name in masks:
            module.forward = MaskedForward(module, masks[name].to(device))
    
    # Step 5: Distillation on retain data to preserve performance
    print("Phase 4: Selective knowledge distillation...")
    
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    temperature = 3.0  # Temperature for softer probability distribution
    alpha = 0.8  # Weight for distillation loss
    
    for epoch in range(4):
        for batch in tqdm(retain_train_dl, desc=f"Distillation Epoch {epoch+1}"):
            inputs, _, targets = [tensor.to(device) for tensor in batch]
            optimizer.zero_grad()
            
            # Forward passes
            student_outputs = model(inputs)
            with torch.no_grad():
                teacher_outputs = unlearning_teacher(inputs)
            
            # Regular cross-entropy loss
            ce_loss = criterion(student_outputs, targets)
            
            # Knowledge distillation loss
            # Only distill knowledge for non-forgotten classes
            mask = (targets != forget_class).float().unsqueeze(1)
            
            # Apply temperature softmax
            soft_targets = F.softmax(teacher_outputs / temperature, dim=1)
            soft_outputs = F.log_softmax(student_outputs / temperature, dim=1)
            
            # Compute masked distillation loss
            distill_loss = F.kl_div(
                soft_outputs, 
                soft_targets, 
                reduction='none'
            ).sum(dim=1) * mask.squeeze()
            
            distill_loss = (temperature ** 2) * distill_loss.mean()
            
            # Combined loss - don't distill for forgotten class
            loss = (1 - alpha) * ce_loss + alpha * distill_loss
            
            loss.backward()
            optimizer.step()
    
    # Step 6: Final optimization - gradient ascent on forget class
    print("Phase 5: Targeted gradient ascent on forget data...")
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    
    # Perform gradient ascent on forget data (maximize loss)
    model.train()
    for epoch in range(2):  # Few epochs of unlearning
        for batch in tqdm(forget_train_dl, desc=f"Targeted Unlearning Epoch {epoch+1}"):
            inputs, _, targets = [tensor.to(device) for tensor in batch]
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = -criterion(outputs, targets)  # Negative sign for ascent
            loss.backward()
            optimizer.step()
    
    # Step 7: Final fine-tuning on retain data
    print("Phase 6: Final fine-tuning...")
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0008, momentum=0.9)
    model.train()
    
    for epoch in range(3):  # Brief fine-tuning
        for batch in tqdm(retain_train_dl, desc=f"Fine-tuning Epoch {epoch+1}"):
            inputs, _, targets = [tensor.to(device) for tensor in batch]
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
    # Restore original forward methods
    for name, module in model.named_modules():
        if name in masks and hasattr(module.forward, 'original_forward'):
            module.forward = module.forward.original_forward
    
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

def NegativeKnowledgeTransferUnlearning(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    forget_class,
    num_classes,
    device,
    **kwargs,
):
    """
    Negative Knowledge Transfer Unlearning.

    This method unlearns specific target classes by:
    1. Training a negative model to *misclassify* forget-class data.
    2. Transferring the 'anti-knowledge' (negative weights) to the base model.
    3. Recovering accuracy on retained classes via knowledge distillation.
    4. Final fine-tuning and adversarial reinforcement on forget data.

    Args:
        model: The original model to be unlearned.
        unlearning_teacher: A frozen copy for knowledge distillation.
        *_dl: DataLoaders for different data partitions.
        forget_class: Class label(s) to forget.
        num_classes: Total number of classes.
        device: Torch device.
    """
    import torch.nn.functional as F
    from torch.optim.lr_scheduler import StepLR

    print("=== Phase 1: Negative Knowledge Acquisition ===")

    negative_model = copy.deepcopy(model).to(device)
    original_state = copy.deepcopy(model.state_dict())
    criterion = nn.CrossEntropyLoss()

    def create_incorrect_targets(targets):
        incorrect = targets.clone()
        for i, true_label in enumerate(targets):
            options = list(range(num_classes))
            options.remove(true_label.item())
            incorrect[i] = random.choice(options)
        return incorrect

    optimizer = torch.optim.SGD(negative_model.parameters(), lr=0.01, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.7)

    negative_model.train()
    for epoch in range(5):
        for inputs, _, targets in tqdm(forget_train_dl, desc=f"NegTrain {epoch+1}"):
            inputs, targets = inputs.to(device), targets.to(device)
            incorrect = create_incorrect_targets(targets)

            optimizer.zero_grad()
            outputs = negative_model(inputs)
            loss = criterion(outputs, incorrect)
            loss.backward()
            optimizer.step()
        scheduler.step()

    print("=== Phase 2: Transferring Anti-Knowledge ===")

    negative_delta = {
        k: negative_model.state_dict()[k] - original_state[k]
        for k in model.state_dict()
        if k in negative_model.state_dict()
    }

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in negative_delta:
                param.data += 0.7 * negative_delta[name]

    print("=== Phase 3: Forget Accuracy Evaluation ===")

    model.eval()
    total, correct, loss_total = 0, 0, 0.0
    with torch.no_grad():
        for inputs, _, targets in forget_valid_dl:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss_total += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

    print(f"Forget Class Accuracy After Transfer: {100. * correct / total:.2f}%")

    print("=== Phase 4: Knowledge Distillation for Retain Classes ===")

    unlearning_teacher.load_state_dict(original_state)
    unlearning_teacher.eval()

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.7)

    alpha, T = 0.9, 4.0

    for epoch in range(4):
        for inputs, _, targets in tqdm(retain_train_dl, desc=f"Distill {epoch+1}"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            student_out = model(inputs)
            with torch.no_grad():
                teacher_out = unlearning_teacher(inputs)

            ce_loss = criterion(student_out, targets)

            soft_t = F.softmax(teacher_out / T, dim=1)
            soft_s = F.log_softmax(student_out / T, dim=1)
            mask = (targets != forget_class).float().unsqueeze(1)

            distill_loss = F.kl_div(soft_s, soft_t, reduction='none').sum(dim=1)
            distill_loss = (T ** 2) * (distill_loss * mask.squeeze()).mean()

            loss = (1 - alpha) * ce_loss + alpha * distill_loss
            loss.backward()
            optimizer.step()
        scheduler.step()

    print("=== Phase 5: Final Retain Fine-tuning ===")

    optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)

    model.train()
    for epoch in range(2):
        for inputs, _, targets in tqdm(retain_train_dl, desc=f"FineTune {epoch+1}"):
            inputs, targets = inputs.to(device), targets.to(device)
            mask = targets != forget_class
            if not torch.any(mask): continue
            inputs, targets = inputs[mask], targets[mask]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    print("=== Phase 6: Adversarial Reinforcement on Forget Data ===")

    model.train()
    for inputs, _, targets in tqdm(forget_train_dl, desc="Adversarial"):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = -criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    print("=== Final Evaluation ===")
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

