import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os
from datetime import datetime
import glob
from data import ContrastiveProjectionDataset 

class ContrastiveModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super(ContrastiveModel, self).__init__()
        
        # Load ResNet50 model
        self.resnet = resnet50(pretrained=False)

        # modify first conv to have input dim of 1
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Modify the final layer to output the desired embedding dimension
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embedding_dim)

    def forward(self, x):
        return self.resnet(x)

class ContrastiveLearningModule(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3):
        super(ContrastiveLearningModule, self).__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.val_embs = []

    def forward(self, x):
        embeddings = self.model(x)
        return F.normalize(embeddings, p=2, dim=1)

    def training_step(self, batch, batch_idx):
        images, _, _ = batch
        # Reshape to process all images at once (B,3,H,W) -> (B*3,1,H,W)
        batch_size = images.shape[0]
        all_embeddings = self(images.view(-1, 1, *images.shape[2:]))
        # Reshape back to separate anchor/positive/negative (B*3,E) -> (B,3,E)
        all_embeddings = all_embeddings.view(batch_size, 3, -1)
        anchor_embeddings, positive_embeddings, negative_embeddings = all_embeddings.unbind(dim=1)

        # Contrastive loss
        # Contrastive loss
        pos_mse = (anchor_embeddings - positive_embeddings).pow(2).sum(dim=-1)
        neg_dist = F.pairwise_distance(anchor_embeddings, negative_embeddings)

        loss = (pos_mse + F.relu(1 - neg_dist)**2).mean()

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, _, ids = batch
        # Reshape to process all images at once (B,3,H,W) -> (B*3,1,H,W)
        batch_size = images.shape[0]
        all_embeddings = self(images.view(-1, 1, *images.shape[2:]))
        # Reshape back to separate anchor/positive/negative (B*3,E) -> (B,3,E)
        all_embeddings = all_embeddings.view(batch_size, 3, -1)
        anchor_embeddings, positive_embeddings, negative_embeddings = all_embeddings.unbind(dim=1)

        # Contrastive loss
        pos_mse = (anchor_embeddings - positive_embeddings).pow(2).sum(dim=-1)
        neg_dist = F.pairwise_distance(anchor_embeddings, negative_embeddings)

        loss = (pos_mse + F.relu(1 - neg_dist)**2).mean()

        self.log('val_loss', loss)

        self.val_embs.append((anchor_embeddings, ids[:,0]))

        return loss

    def on_validation_epoch_end(self):
        all_embeddings = torch.cat([x[0] for x in self.val_embs])
        all_labels = torch.cat([x[1] for x in self.val_embs])

        # Get unique object IDs and select one embedding per object
        unique_labels, indices = torch.unique(all_labels, return_inverse=True)
        query_mask = torch.zeros_like(all_labels, dtype=torch.bool)
        for i in range(len(unique_labels)):
            # Get first occurrence of each label
            query_mask[torch.where(indices == i)[0][0]] = True
            
        query_embeddings = all_embeddings[query_mask]
        query_labels = all_labels[query_mask]

        # Normalize embeddings
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        all_embeddings_norm = F.normalize(all_embeddings, p=2, dim=1)

        batch_size = 32  # Process in batches to save memory
        top_k = 32
        all_top_indices = []
        
        # Compute similarities in batches
        for i in range(0, len(query_embeddings), batch_size):
            batch_queries = query_embeddings[i:i+batch_size]
            batch_similarities = torch.mm(batch_queries, all_embeddings_norm.t())
            _, batch_top_indices = torch.topk(batch_similarities, k=top_k, dim=1)
            all_top_indices.append(batch_top_indices)
            
        top_indices = torch.cat(all_top_indices, dim=0)
        top_labels = all_labels[top_indices]
        
        # Compare with query labels
        matches = (top_labels == query_labels.unsqueeze(1))
        
        # Calculate fraction of matches (excluding self-match)
        match_frac = (matches[:,1:].float().mean(dim=1)).mean()
        
        self.log('val_match_fraction', match_frac)
        self.val_embs = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=10000,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Contrastive Learning with PyTorch Lightning')
    parser.add_argument('--exp_name', type=str, help='Name of the experiment')
    parser.add_argument('--resume_run', type=str, help='Full experiment name to resume (e.g., "my_experiment_20240315_123456")')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of the embedding space')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--log_dir', type=str, default='/home/gridsan/jroney/cryofinder-training', help='Directory to save logs and checkpoints')
    args = parser.parse_args()


    # Create or load experiment name
    if args.resume_run:
        exp_name = args.resume_run
        print(f"Resuming training from experiment: {exp_name}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{args.exp_name}_{timestamp}"
        log_dir = os.path.join(args.log_dir, exp_name)
        os.makedirs(log_dir, exist_ok=True)

    print(f"Starting new experiment: {exp_name}")

    # Check for existing checkpoints
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    latest_checkpoint=None
    if os.path.exists(checkpoint_dir):
        existing_checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))
        if existing_checkpoints:
            latest_checkpoint=max(existing_checkpoints, key=os.path.getmtime)

    # Setup logging
    logger = TensorBoardLogger(
        save_dir=log_dir,
        name='tensorboard',
        default_hp_metric=False
    )

    # Setup checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(log_dir, 'checkpoints'),
        filename='{epoch:02d}-{val_match:.2f}',
        save_top_k=1,
        monitor='val_match_fraction',
        mode='max',
        save_last=True
    )


    # Load your dataset
    train_dat = torch.load("/home/gridsan/jroney/train_projections.pt")
    val_dat = torch.load("/home/gridsan/jroney/val_projections.pt")

    train_dataset = ContrastiveProjectionDataset(train_dat['images'], train_dat['phis'], train_dat['thetas'], train_dat['ids'], pos_angle_threshold=45)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: ContrastiveProjectionDataset.collate_fn(x, train_dataset.lat, train_dataset.mask, train_dataset.freqs, ctf_corrupt=False, noise=True))

    val_dataset = ContrastiveProjectionDataset(val_dat['images'], val_dat['phis'], val_dat['thetas'], val_dat['ids'])
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size*10, shuffle=False)

    # Initialize model and training module
    model = ContrastiveModel(embedding_dim=args.embedding_dim)
    training_module = ContrastiveLearningModule(model)

    # Train the model
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        devices=1,
        accelerator='gpu',
        gradient_clip_val=10.0  # Add gradient clipping
    )

    trainer.fit(training_module, train_loader, val_loader, ckpt_path=latest_checkpoint)
