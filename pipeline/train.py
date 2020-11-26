import sys
from itertools import chain

from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.append('../src/')
from moner.data import NERMultiOutput
from moner.model import MultiOutputNER
from moner.utils import get_trues_pred_lists
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from torch.nn import Linear, ModuleList, LogSoftmax, NLLLoss
import torch
from torch.optim import Adam
import pdb
from tqdm import tqdm

MODEL_NAME_OR_PATH = 'bert-base-cased'
N_CLASSES = 14
TRAIN_FILENAME = '../data/train.txt'
DEV_FILENAME = '../data/dev.txt'

print(f"Loading Tokenizer for mode:l {MODEL_NAME_OR_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
print(f"Initializing model with: {MODEL_NAME_OR_PATH}")
model = MultiOutputNER(N_CLASSES, MODEL_NAME_OR_PATH)

print("Loading training data:")
train = NERMultiOutput(TRAIN_FILENAME, num_labels=N_CLASSES, tokenizer=tokenizer)
print("Loading dev data:")
dev = NERMultiOutput(DEV_FILENAME, num_labels=N_CLASSES, tokenizer=tokenizer, label2idx=train.label2idx)

out = dev[1]
for k, v in out.items():
    print(k, len(v))
    
print("Materializing training data")
train_dataloader = [i for i in DataLoader(train, batch_size=4, shuffle=False, num_workers=0)]
print("Materializing dev data")
val_dataloader = [i for i in DataLoader(dev, batch_size=4, shuffle=False, num_workers=0)]

#print("Fetching and running batch")
#batch = train_dataloader[0]
#preds, loss = model(batch)
print("Creating optimizer")
optimizer = Adam(params = model.parameters())

it = 0
total_loss = 0
stop_at = 50
all_preds = []
all_trues = []
for cls in range(model.n_classes):
    all_preds.append([])
    all_trues.append([])
print("Starting training loop")
for batch in tqdm(train_dataloader):
    #print(it)
    model.zero_grad()
    it += 1
    preds, loss = model(batch)
    total_loss += loss

    trues, preds = get_trues_pred_lists(batch, preds)

    for cls in range(model.n_classes):
        for b in range(len(trues)):
            all_trues[cls].extend(trues[b][cls])
            all_preds[cls].extend(preds[b][cls])

    #print(total_loss / it)
    loss.backward()

    optimizer.step()

#true_label_lists = [[ll.numpy().tolist() for ll, aa in zip(l, a) if aa] for l, a in zip(batch['labels'], batch['attention_mask'])]
#pred_label_lists = [[ll.numpy().tolist() for ll, aa in zip(l, a) if aa] for l, a in zip(torch.argmax(preds, dim=3), batch['attention_mask'])]
#trues = [list(zip(*l)) for l in true_label_lists]
#preds = [list(zip(*l)) for l in pred_label_lists]

all_preds = []
all_trues = []
for cls in range(model.n_classes):
    all_preds.append([])
    all_trues.append([])
print("Starting eval loop")
for batch in tqdm(val_dataloader):
    #print(it)
    model.zero_grad()
    it += 1
    preds, loss = model(batch)
    total_loss += loss

    trues, preds = get_trues_pred_lists(batch, preds)

    for cls in range(model.n_classes):
        for b in range(len(trues)):
            all_trues[cls].extend(trues[b][cls])
            all_preds[cls].extend(preds[b][cls])

    #print(total_loss / it)

for cls in range(model.n_classes):
    baseline_accuracy = sum([train.idx2label[cls][i] == train.OUTSIDE for i in all_trues[cls]])/len(all_trues[cls])
    print(baseline_accuracy, precision_score(all_trues[cls], all_preds[cls], average='micro'))

pdb.set_trace()
