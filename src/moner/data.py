from torch.utils.data import Dataset, DataLoader
from itertools import chain

class NERMultiOutput(Dataset):
    """NER Multi Output Dataset."""

    def __init__(self, 
                filepath, 
                sep='\t', 
                num_labels=1, 
                label_scheme='BILUO'
        ):
        """
        Args:
            filepath (string): Path to CONLL2003 formatted sequence tagging dataset file.
            sep (string, optional): Character separating tokens and labels
            n_labels (int, optional): Number of labels in the dataset 
            label_scheme (string, optional): Sequential labeling scheme
        """
        self.filepath = filepath
        self.sep = sep
        self.num_labels = num_labels
        self.label_scheme = label_scheme
        self.OUTSIDE = self.label_scheme[-1]

        self.tokens, self.labels = self._parse_conll()

    def _parse_conll(self):
        """
        Parse the raw text of a CONLL2003 formatted seq dataset
        """
        raw = open(self.filepath).read().split('\n\n')
        #raw = [i.strip('"') for i in raw]
        lines = [[i.split(self.sep) for i in r.split('\n') if i] for r in raw]
        lines = [[i for i in j if len(i[1:]) == self.num_labels] for j in lines]
        tokens = [[i[0] for i in j] for j in lines]
        labels = [[i[1:] for i in j] for j in lines]

        return tokens, labels

    def _get_label_indexes(self):

        self.label2idx = {}

        complete_labels = list(zip(*chain.from_iterable(self.labels)))
        for it, channel in enumerate(complete_labels):
            # Dumb fix for null and '-' showing up in labels
            channel = (i for i in set(channel) if i not in ['', '-'])
            self.label2idx[it] = {label: i for i, label in enumerate(channel)}

        self.idx2label = {
                    k: {vv:kk for kk, vv in v.items()}
                    for k, v in self.label2idx.items()
                }

    def _encode_label_indexes(self):

        self.labels_encoded = []

        for l in self.labels:
            l_out = []
            channel_labels = zip(*l)
            for channel_it, ll in enumerate(channel_labels):
                lkp = self.label2idx[channel_it]
                ll = [lll if lll in lkp.keys() else self.OUTSIDE for lll in ll]
                ll = [lkp[lll] for lll in ll]

                l_out.append(ll)
            l_out = list(zip(*l_out))

            self.labels_encoded.append(l_out)

    

    def __len__(self):
        #return len(self.landmarks_frame)
        pass
    def __getitem__(self, idx):
        #if torch.is_tensor(idx):
        #    idx = idx.tolist()

        #img_name = os.path.join(self.root_dir,
        #                        self.landmarks_frame.iloc[idx, 0])
        #image = io.imread(img_name)
        #landmarks = self.landmarks_frame.iloc[idx, 1:]
        #landmarks = np.array([landmarks])
        #landmarks = landmarks.astype('float').reshape(-1, 2)
        #sample = {'image': image, 'landmarks': landmarks}

        #if self.transform:
        #    sample = self.transform(sample)

        #return sample
        pass

