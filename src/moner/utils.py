import torch


def get_trues_pred_lists(batch, preds):
    true_label_lists = [[ll.numpy().tolist() for ll, aa in zip(l, a) if aa] for l, a in zip(batch['labels'], batch['attention_mask'])]
    pred_label_lists = [[ll.numpy().tolist() for ll, aa in zip(l, a) if aa] for l, a in zip(torch.argmax(preds, dim=3), batch['attention_mask'])]
    trues = [list(zip(*l)) for l in true_label_lists]
    preds = [list(zip(*l)) for l in pred_label_lists]
    return trues, preds