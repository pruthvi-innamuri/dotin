import torch as th
import pytorch_lightning as pl
import datasets
import numpy
import pandas
import transformers
from absl import app, flags, logging


def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()    
    keys_list = [keys for keys in flags_dict]    
    for keys in keys_list:
        FLAGS.__delattr__(keys)

del_all_flags(flags.FLAGS)

debugVar = False
epochsVar = 10
batch_sizeVar = 8
lrVar = 1e-2
momentumVar = 0.9
modelVar = 'bert-base-uncased'
seq_lengthVar = 32
percentVar = 5
flags.DEFINE_boolean('debug', False, '')
flags.DEFINE_integer('epochs', 10, '')
flags.DEFINE_integer('batch_size', 8, '')
flags.DEFINE_float('lr', 1e-2, '')
flags.DEFINE_float('momentum', .9, '')
flags.DEFINE_string('model', 'bert-base-uncased', '')
flags.DEFINE_integer('seq_length', 32, '')
flags.DEFINE_integer('percent', 5, '')

FLAGS = flags.FLAGS

class sentimentClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = transformers.BertForSequenceClassification.from_pretrained(modelVar)
        self.loss = th.nn.CrossEntropyLoss(reduction='none')

    def prepare_data(self):
        tokenizer = transformers.BertTokenizer.from_pretrained(modelVar)

        def _tokenize(x):
            x['input_ids'] = tokenizer.batch_encode_plus(
                    x['text'], 
                    max_length=seq_lengthVar, 
                    pad_to_max_length=True)['input_ids']
            return x

        def _prepare_ds(split):
            ds = datasets.load_dataset('imdb', split=f'{split}[:{batch_sizeVar if debugVar else f"{percentVar}%"}]')
            ds = ds.map(_tokenize, batched=True)
            ds.set_format(type='torch', columns=['input_ids', 'label'])
            return ds

        self.train_ds, self.test_ds = map(_prepare_ds, ('train', 'test'))

    def forward(self, input_ids):
        mask = (input_ids != 0).float()
        logits, = self.model(input_ids, mask)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch['input_ids'])
        loss = self.loss(logits, batch['label']).mean()
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch['input_ids'])
        loss = self.loss(logits, batch['label'])
        acc = (logits.argmax(-1) == batch['label']).float()
        return {'loss': loss, 'acc': acc}

    def validation_epoch_end(self, outputs):
        loss = th.cat([o['loss'] for o in outputs], 0).mean()
        acc = th.cat([o['acc'] for o in outputs], 0).mean()
        out = {'val_loss': loss, 'val_acc': acc}
        return {**out, 'log': out}

    def train_dataloader(self):
        return th.utils.data.DataLoader(
                self.train_ds,
                batch_size=batch_sizeVar,
                drop_last=True,
                shuffle=True,
                )

    def val_dataloader(self):
        return th.utils.data.DataLoader(
                self.test_ds,
                batch_size=batch_sizeVar,
                drop_last=False,
                shuffle=True,
                )

    def configure_optimizers(self):
        return th.optim.SGD(
            self.parameters(),
            lr=lrVar,
            momentum=momentumVar,
        )


def main():
    model = sentimentClassifier()
    trainer = pl.Trainer(
        default_root_dir='logs',
        gpus=(1 if th.cuda.is_available() else 0),
        max_epochs=epochsVar,
        fast_dev_run=debugVar,
        logger=pl.loggers.TensorBoardLogger('logs/', name='imdb', version=0),
    )
    trainer.fit(model)

if __name__ == "__main__":
    main()
