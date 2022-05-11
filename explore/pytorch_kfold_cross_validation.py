from torch.utils.data import DataLoader,SubsetRandomSampler, ConcatDataset
from sklearn.model_selection import KFold
import numpy as np

# 대부분의 function들은 module화 처리했고, config파일 같은 경우는 가져오지 않고 코드 흐름만 보여주기 위해 진행  -> 이러한 흐름이다라는 부부만 기억해서 나중에 적용하도록 하기 
# 3D CNN model 
def train_epoch(model, criterion, optimizer, dataloader):
    train_correct = 0
    train_samples = 0
    losses = []
    # loss를 초기화해주는 코드 이전 fold에서 학습한 그거를 초기화시켜주는 
    #for epoch in range(NUM_EPOCH):
    for data, targets in dataloader:
        data = data.to(device=device, non_blocking=True)
        targets = targets.to(device=device, non_blocking=True)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        losses.append(loss.item())

        # backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        _, predictions = scores.max(1) 
        # scores, predictions = torch.max(scores.data, 1)
        train_correct += (predictions == targets).sum()
        train_samples += predictions.size(0)
        
    print(f"Cost at train epoch is {sum(losses)/len(losses):.5f}")  
    return  roc_auc_score(targets.cpu().numpy(), predictions.cpu().numpy())

"""train epoch 와 다른점은 valid 부분은 loss.backward(), optimizer.step()을 진행하지 않은 부분애서 차이 확인 가능!"""
def valid_epoch(model, criterion, optimizer, dataloader):
    val_correct = 0
    val_samples = 0
    losses = []
   # for epoch in range(NUM_EPOCH):
    for data, targets in dataloader:
        data = data.to(device=device, non_blocking=True)
        targets = targets.to(device=device, non_blocking=True)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        losses.append(loss.item())

        # backward
        optimizer.zero_grad(set_to_none=True)

        _, predictions = scores.max(1) 
        # scores, predictions = torch.max(scores.data, 1)
        val_correct += (predictions == targets).sum()
        val_samples += predictions.size(0)

    print(f"Cost at valid epoch is {sum(losses)/len(losses):.5f}")  
    return  roc_auc_score(targets.cpu().numpy(), predictions.cpu().numpy())






def main():
    k = 5
    splits = KFold(n_splits = k , shuffle = True, random_state = 42)
    foldperf = {}
    foldperf2 = {}
    train_dataset, valid_dataset, test_dataset = get_kfold_dataset(X_TRAIN_PATH, Y_TRAIN_PATH, X_VALID_PATH, Y_VALID_PATH, X_TEST_PATH, Y_TEST_PATH, BATCH_SIZE)
    dataset = ConcatDataset([train_dataset, valid_dataset])

    device = get_device() # module


    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(dataset)))):
        print(f'Fold {fold + 1}')

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(dataset, batch_size = BATCH_SIZE, sampler = train_sampler)
        valid_loader = DataLoader(dataset, batch_size = BATCH_SIZE, sampler = valid_sampler)

        history = {'train_roc':[],'valid_roc':[]}

        model = create_model(device) # module
        optimizer = get_optim(model, LEARNING_RATE, WEIGHT_DECAY,ADAM_EPSILON) # module
        criterion = get_loss() # module

        for epoch in range(NUM_EPOCH):

            train_correct=train_epoch(model, criterion, optimizer, train_loader)
            valid_correct=valid_epoch(model, criterion, optimizer, valid_loader)

            train_roc = train_correct * 100
            valid_roc = valid_correct * 100

            print("Epoch:{}/{} AVG Training Acc {:.2f} % AVG Valid Acc {:.2f} %".format(epoch + 1, NUM_EPOCH,train_roc,valid_roc))  

            history['train_roc'].append(train_roc)
            history['valid_roc'].append(valid_roc)
        # history['test_acc'].append(test_acc)


    for fold, (test_idx) in enumerate(splits.split(np.arange(len(test_dataset)))):
        print(f'Fold {fold + 1}')

        test_sampler = SubsetRandomSampler(test_idx)
        test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, sampler = test_sampler)

        history2 = {'test_roc':[]}

        model = model
        optimizer = get_optim(model, LEARNING_RATE, WEIGHT_DECAY,ADAM_EPSILON)
        criterion = get_loss()

        for epoch in range(NUM_EPOCH):

            test_correct=valid_epoch(model, criterion, optimizer, valid_loader)

            test_roc = test_correct * 100

            print("Epoch:{}/{}  AVG Test Roc {:.2f} %".format(epoch + 1, NUM_EPOCH,test_roc))  

            history2['test_roc'].append(test_roc)

    torch.save(model,'model/kfold/k_cross_CNN1.pt')  

    train_f, val_f, test_f = [],[], []
    k=5
    for f in range(1,k+1):
        train_f.append(np.mean(foldperf[f'fold{f}']['train_roc']))
        val_f.append(np.mean(foldperf[f'fold{f}']['valid_roc']))
        test_f.append(np.mean(foldperf2[f'fold{f}']['test_roc']))

    print(f'Performance of {k} fold cross validation')
    print("Average Training Roc: {:.2f} \t Average Valid Roc: {:.2f} \t Average Test Roc: {:.2f}".format(np.mean(train_f),np.mean(val_f),np.mean(test_f)))  

