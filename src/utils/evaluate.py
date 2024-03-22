

import torch


def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    logits = []
    targets = []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            output = model(X_batch.to(device)).cpu()
            logits.append(output)
            targets.append(y_batch)
    
    logits = torch.cat(logits)
    targets = torch.cat(targets)
    loss = loss_fn(logits, targets).mean().item()

    pred = logits.max(1)[1]
    #pred = logits.argmax(dim=1)
    acc = (pred == targets).sum().item() / len(targets)
    return loss, acc

def epoch_test(clf, loader,  criterion, device):
    # BEGIN Solution (do not delete this comment!)
    ret_loss, correct = (0.0, 0)
    clf.eval()
    
    with torch.no_grad():
        for x, labels in loader:
            x, labels = x.to(device), labels.to(device)
            
            all_pred = clf(x)
            loss = criterion(all_pred, labels)
            
            ret_loss += loss.item()
            pred = all_pred.max(1)[1]

            correct += (pred==labels).sum().item()

    ret_loss = ret_loss / len(loader)
    accuracy = correct / len(loader.dataset)

    return ret_loss, accuracy