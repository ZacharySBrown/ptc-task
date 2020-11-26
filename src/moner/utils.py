import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

use_cuda = torch.cuda.is_available()

def get_trues_pred_lists(batch, preds):
    if use_cuda:
        true_label_lists = [[ll.cpu().numpy().tolist() for ll, aa in zip(l, a) if aa] for l, a in zip(batch['labels'], batch['attention_mask'])]
        pred_label_lists = [[ll.cpu().numpy().tolist() for ll, aa in zip(l, a) if aa] for l, a in zip(torch.argmax(preds, dim=3), batch['attention_mask'])]
    else:
        true_label_lists = [[ll.numpy().tolist() for ll, aa in zip(l, a) if aa] for l, a in zip(batch['labels'], batch['attention_mask'])]
        pred_label_lists = [[ll.numpy().tolist() for ll, aa in zip(l, a) if aa] for l, a in zip(torch.argmax(preds, dim=3), batch['attention_mask'])]
    trues = [list(zip(*l)) for l in true_label_lists]
    preds = [list(zip(*l)) for l in pred_label_lists]
    return trues, preds
