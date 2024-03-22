

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
    acc = (logits.argmax(dim=1) == targets).sum().item() / len(targets)
    return loss, acc