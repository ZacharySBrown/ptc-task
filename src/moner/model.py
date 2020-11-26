from transformers import AutoModel
from torch.nn import Linear, ModuleList, LogSoftmax, NLLLoss
import torch

class MultiOutputNER(torch.nn.Module):
    def __init__(self, n_classes, MODEL_NAME_OR_PATH, labels_per_class=5, freeze_transformer=True):
        """
        ...
        """
        super(MultiOutputNER, self).__init__()
        self.bert_model = AutoModel.from_pretrained(MODEL_NAME_OR_PATH)
        self.freeze_transformer = freeze_transformer
        if self.freeze_transformer:
            for param in self.bert_model.parameters():
                param.requires_grad = False

        #labels_per_class = 5
        #hidden_dim = bert_model.config.hidden_size
        self.hidden_dim = self.bert_model.config.hidden_size
        self.n_classes = n_classes
        self.labels_per_class = labels_per_class # defaults to 5 for BIOLU per class

        self.loss_function = NLLLoss()
        self.sm = LogSoftmax(dim=2)
        self.classifiers = ModuleList([Linear(self.hidden_dim, labels_per_class) for _ in range(self.n_classes)])

    def forward(self, batch):
        """
        ...
        """

        model_inputs = {k: v for k, v in batch.items() if k != 'labels'}
        labels = batch['labels']
        attention_mask = model_inputs['attention_mask']

        masked_labels = torch.stack(
            [
                torch.where(attention_mask != 0, labels[:,:,i], -100) 
                for i in range(self.n_classes)
            ], 
        dim=2
        )

        #preds = torch.stack([sm(cls(output[0])) for cls in classifiers]).permute(1,2,0,3)
        output = self.bert_model(**model_inputs)
        preds = [self.sm(cls(output[0])) for cls in self.classifiers]
        
        losses = torch.cat(
            [
                self.loss_function(preds[i].transpose(1,2), masked_labels[:,:,i]).unsqueeze(0)
                for i in range(self.n_classes)
            ]
        )

        preds = torch.stack(preds, dim=2)

        loss_val = losses.sum()
        
        return preds, loss_val