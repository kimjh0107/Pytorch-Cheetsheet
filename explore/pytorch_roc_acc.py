import torch
from sklearn.metrics import roc_auc_score

def calculate_roc(loader, model, device):
    num_correct = 0
    num_samples = 0
    model.eval() 

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device = device)
            y = y.to(device = device)
            #x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)  
            num_correct += (predictions == y).sum() 
            num_samples += predictions.size(0)  

    model.train()
    #return num_correct/num_samples
    return roc_auc_score(y.cpu().numpy(), predictions.cpu().numpy())
