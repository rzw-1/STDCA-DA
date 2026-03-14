import os
import scipy.io as sio
import torch
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import random
import torch.nn as nn
from models import STDCA_Net

def load_train_val_test_data(file_list, folder):
    X, y = [], []
    for filename in file_list:
        data_dict = sio.loadmat(str(os.path.join(folder, filename)))
        data = data_dict['data']
        label = data_dict['label']

        data = np.transpose(data, (2, 0, 1))
        data = (data - np.mean(data, axis=(1, 2), keepdims=True)) / \
               (np.std(data, axis=(1, 2), keepdims=True) + 1e-8)

        X.append(torch.tensor(data, dtype=torch.float32))
        y.append(torch.tensor(label.flatten(), dtype=torch.long))

    return torch.cat(X), torch.cat(y)

def load_target_data(file_list, folder):
    X = []
    for filename in file_list:
        data_dict = sio.loadmat(str(os.path.join(folder, filename)))
        data = data_dict['data']

        data = np.transpose(data, (2, 0, 1))
        data = (data - np.mean(data, axis=(1, 2), keepdims=True)) / \
               (np.std(data, axis=(1, 2), keepdims=True) + 1e-8)

        X.append(torch.tensor(data, dtype=torch.float32))

    return torch.cat(X)


def create_dataloaders(X_source, y_source, X_target, X_val, y_val, X_test, y_test, batch_size):
    source_domains = torch.zeros(len(X_source), dtype=torch.long)
    target_domains = torch.ones(len(X_target), dtype=torch.long)

    train_dataset = TensorDataset(
        torch.cat([X_source, X_target]),
        torch.cat([y_source,
                   torch.full((len(X_target),), -100, dtype=torch.long)]), 
        torch.cat([source_domains, target_domains])
    )

    val_dataset = TensorDataset(X_val, y_val)

    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader

def train_epoch(model, train_loader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_aux_loss = 0.0
    total_dom_loss = 0.0
    correct = 0
    total = 0

    alpha = 2. / (1. + np.exp(-10 * epoch / total_epochs)) - 1

    for x, y, domain_y in train_loader:
        x, y, domain_y = x.to(device), y.to(device), domain_y.to(device)

        optimizer.zero_grad()
        class_pred, aux_pred, domain_pred = model(x, alpha)

        cls_loss = criterion(class_pred, y)
        aux_loss = criterion(aux_pred, y)

        dom_loss = criterion(domain_pred, domain_y)

        loss = cls_loss + 0.5 * aux_loss + 0.3 * dom_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_aux_loss += aux_loss.item()
        total_dom_loss += dom_loss.item()

        combined_probs = (F.softmax(class_pred, 1) + F.softmax(aux_pred, 1)) / 2
        _, predicted = torch.max(combined_probs.data, 1)

        total += y.size(0)
        correct += (predicted.eq(y)).sum().item()

    avg_loss = total_loss / len(train_loader)
    avg_cls_loss = total_cls_loss / len(train_loader)
    avg_aux_loss = total_aux_loss / len(train_loader)
    accuracy = correct / total

    print(f"Epoch {epoch + 1}/{total_epochs} | "
          f"Train Acc: {accuracy:.2%} | "
          f"Loss: {avg_loss:.4f} | "
          f"Cls Loss: {avg_cls_loss:.4f} | "
          f"Aux Loss: {avg_aux_loss:.4f} | "
          f"Dom Loss: {total_dom_loss / len(train_loader):.4f} | "
          f"λ: {alpha:.4f}"
          )

    return accuracy, avg_loss

def evaluate(model, test_loader, device, fold=None):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            class_pred, aux_pred, _ = model(x, alpha=0.0)

            combined_probs = (F.softmax(class_pred, 1) + F.softmax(aux_pred, 1)) / 2
            _, predicted = torch.max(combined_probs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    TN, FP, FN, TP = cm.ravel()
    Acc = (TP + TN) / (TP + TN + FP + FN)
    Sens = TP / (TP + FN)
    Spec = TN / (TN + FP)
    Prec = TP / (TP + FP)
    Fl = 2 * TP / (2 * TP + FP + FN)

    return {
        'accuracy': Acc,
        'sensitivity': Sens,
        'specificity': Spec,
        'precision': Prec,
        'f1_score': Fl,
        'confusion_matrix': cm
    }, Acc

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed(24)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"   
    device = torch.device("cuda:0")

    print(f"Using device: {device}")

    num_channels = 19
    time_length = 1024
    batch_size = 32
    num_classes = 2
    embed_size = 128

    hc_folder = './MPHC_pre_4S/HC'
    mdd_folder = './MPHC_pre_4S/MDD'

    hc_files = [f for f in os.listdir(hc_folder) if f.endswith('.mat')]
    mdd_files = [f for f in os.listdir(mdd_folder) if f.endswith('.mat')]

    fold_results = []
    fold_accuracies = []

    kf = KFold(n_splits=10, shuffle=True, random_state=24)

    hc_splits = list(kf.split(hc_files))
    mdd_splits = list(kf.split(mdd_files))

    for fold in range(10):
        print(f"\n=== Fold {fold + 1}/{10} ===")

        hc_train_idx, hc_test_idx = hc_splits[fold]
        mdd_train_idx, mdd_test_idx = mdd_splits[fold]

        hc_train_files = [hc_files[i] for i in hc_train_idx]
        hc_test_files = [hc_files[i] for i in hc_test_idx]

        mdd_train_files = [mdd_files[i] for i in mdd_train_idx]
        mdd_test_files = [mdd_files[i] for i in mdd_test_idx]

        hc_source_files, hc_val_files = train_test_split(
            hc_train_files, test_size=0.2, random_state=42
        )

        mdd_source_files, mdd_val_files = train_test_split(
            mdd_train_files, test_size=0.2, random_state=42
        )

        hc_target_files = hc_test_files
        mdd_target_files = mdd_test_files

        hc_source_x, hc_source_y = load_train_val_test_data(hc_source_files, hc_folder)
        mdd_source_x, mdd_source_y = load_train_val_test_data(mdd_source_files, mdd_folder)

        hc_target_x = load_target_data(hc_target_files, hc_folder)
        mdd_target_x = load_target_data(mdd_target_files, mdd_folder)

        hc_x_val, hc_y_val = load_train_val_test_data(hc_val_files, hc_folder)
        mdd_x_val, mdd_y_val = load_train_val_test_data(mdd_val_files, mdd_folder)

        hc_x_test, hc_y_test = load_train_val_test_data(hc_test_files, hc_folder)
        mdd_x_test, mdd_y_test = load_train_val_test_data(mdd_test_files, mdd_folder)

        X_source = torch.cat([hc_source_x, mdd_source_x])
        y_source = torch.cat([hc_source_y, mdd_source_y])

        X_target = torch.cat([hc_target_x, mdd_target_x])

        X_val = torch.cat([hc_x_val, mdd_x_val])
        y_val = torch.cat([hc_y_val, mdd_y_val])

        X_test = torch.cat([hc_x_test, mdd_x_test])
        y_test = torch.cat([hc_y_test, mdd_y_test])

        train_loader, val_loader, test_loader = create_dataloaders(
            X_source, y_source, X_target, X_val, y_val, X_test, y_test, batch_size
        )

        model = STDCA_Net(num_channels, time_length, num_classes, embed_size).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0
        patience = 50
        bad_epoch = 0

        for epoch in range(200):

            train_acc, train_loss = train_epoch(
                model, train_loader, optimizer, criterion, device, epoch, 100)
 
            model.eval()
            val_correct = val_total = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    class_pred, aux_pred, _ = model(x, alpha=0.0)
                    combined_probs = (F.softmax(class_pred, 1) + F.softmax(aux_pred, 1)) / 2
                    val_correct += (combined_probs.argmax(1) == y).sum().item()
                    val_total += y.size(0)

            val_acc = val_correct / val_total
            print(f"Fold {fold + 1} | Epoch {epoch:02d} | Val Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                bad_epoch = 0

                save_path = os.path.join(SAVE_DIR, f"best_fold{fold + 1}.pth")
                torch.save(model.state_dict(), save_path)
            else:
                bad_epoch += 1
                if bad_epoch >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            model.train()

        save_path = os.path.join(SAVE_DIR, f"best_fold{fold + 1}.pth")
        model.load_state_dict(torch.load(save_path, map_location=device))

        test_results, test_accuracy = evaluate(model, test_loader, device, fold + 1)

        fold_results.append(test_results)
        fold_accuracies.append(test_accuracy)

        print(f"\nFold {fold + 1} Detailed Results:")

        print(f"Accuracy: {test_results['accuracy']:.4f}")
        print(f"Sensitivity: {test_results['sensitivity']:.4f}")
        print(f"Specificity: {test_results['specificity']:.4f}")
        print(f"Precision: {test_results['precision']:.4f}")
        print(f"F1 Score: {test_results['f1_score']:.4f}")

        print("Confusion Matrix:")
        print(test_results['confusion_matrix'])

    avg_accuracy = np.mean([r['accuracy'] for r in fold_results])
    avg_sensitivity = np.mean([r['sensitivity'] for r in fold_results])
    avg_specificity = np.mean([r['specificity'] for r in fold_results])
    avg_precision = np.mean([r['precision'] for r in fold_results])
    avg_f1 = np.mean([r['f1_score'] for r in fold_results])

    std_accuracy = np.std([r['accuracy'] for r in fold_results])
    std_sensitivity = np.std([r['sensitivity'] for r in fold_results])
    std_specificity = np.std([r['specificity'] for r in fold_results])
    std_precision = np.std([r['precision'] for r in fold_results])
    std_f1 = np.std([r['f1_score'] for r in fold_results])

    print("\n=== Final Average Results ===")
    print(f"Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Average Sensitivity: {avg_sensitivity:.4f} ± {std_sensitivity:.4f}")
    print(f"Average Specificity: {avg_specificity:.4f} ± {std_specificity:.4f}")
    print(f"Average Precision: {avg_precision:.4f} ± {std_precision:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f} ± {std_f1:.4f}")

if __name__ == "__main__":
    SAVE_DIR = r"./STDCA-DA"
    os.makedirs(SAVE_DIR, exist_ok=True)

    main()
