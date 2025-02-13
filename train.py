import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from pytorch_lightning.callbacks import ModelCheckpoint
from contrastive_projection_dataset import ContrastiveProjectionDataset  # Ensure this is the correct import path

class ContrastiveModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super(ContrastiveModel, self).__init__()
        
        # Load ResNet50 model
        self.resnet = resnet50(pretrained=False)
        # Modify the final layer to output the desired embedding dimension
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embedding_dim)

    def forward(self, x):
        return self.resnet(x)

class ContrastiveLearningModule(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3):
        super(ContrastiveLearningModule, self).__init__()
        self.model = model
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        anchors, positives, negatives = batch
        anchor_embeddings = self(anchors)
        positive_embeddings = self(positives)
        negative_embeddings = self(negatives)

        # Contrastive loss
        pos_dist = F.pairwise_distance(anchor_embeddings, positive_embeddings)
        neg_dist = F.pairwise_distance(anchor_embeddings, negative_embeddings)
        loss = torch.mean(F.relu(pos_dist - neg_dist + 1.0))  # Margin of 1.0

        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

def main():
    # Load your dataset
    dataset = ContrastiveProjectionDataset(...)  # Fill in with appropriate arguments
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=ContrastiveProjectionDataset.collate_fn)

    # Initialize model and training module
    model = ContrastiveModel()
    training_module = ContrastiveLearningModule(model)

    # Set up checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        dirpath='checkpoints/',
        filename='contrastive-{epoch:02d}-{train_loss:.2f}',
        save_top_k=3,
        mode='min',
    )

    # Initialize trainer
    trainer = pl.Trainer(max_epochs=10, callbacks=[checkpoint_callback])

    # Train the model
    trainer.fit(training_module, dataloader)

if __name__ == '__main__':
    main()
