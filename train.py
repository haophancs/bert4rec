import argparse
import os

import torch
from torch.utils.data import DataLoader

from src.data.interaction import InteractionDataset
from src.data.sequential import SequentialItemsDataset
from src.helpers import get_trainer
from src.models.sequential import BERT4Rec


def train(
        data_name,
        data_root,
        rating_csv_file,
        data_user_col,
        data_item_col,
        data_chrono_col,
        device,
        seq_length,
        mask_p,
        hidden_size,
        batch_size,
        epochs,
        lr,
        dropout,
        weight_decay,
        checkpoint_dir,
        log_dir,
        num_workers=10
):
    data_path = os.path.join(data_root, data_name, rating_csv_file)
    print(f'Loading interaction data from {data_path}')
    interaction_data = InteractionDataset(
        df_csv_path=data_path,
        user_id_col=data_user_col,
        item_id_col=data_item_col,
        chrono_col=data_chrono_col,
    )
    print(str(interaction_data))

    train_dataset = SequentialItemsDataset(
        interaction_data=interaction_data, split='train', seq_length=seq_length, mask_p=mask_p
    )
    val_dataset = SequentialItemsDataset(
        interaction_data=interaction_data, split='val', seq_length=seq_length
    )
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers
    )
    model = BERT4Rec(
        seq_length=seq_length,
        vocab_size=train_dataset.vocab_size,
        mask_token=train_dataset.mask_token,
        pad_token=train_dataset.pad_token,
        hidden_size=hidden_size,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay
    )
    trainer = get_trainer(
        epochs=epochs,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        checkpoint_prefix=f"{model.__class__.__name__.lower()}_{data_name}",
        device=device
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training parameters for BERT4Rec.')
    parser.add_argument('--data_name', type=str, default='ml-25m')
    parser.add_argument('--data_root', type=str, default='./datasets')
    parser.add_argument('--rating_csv_file', type=str, default='ratings.csv')
    parser.add_argument('--data_user_col', type=str, default='userId')
    parser.add_argument('--data_item_col', type=str, default='movieId')
    parser.add_argument('--data_chrono_col', type=str, default='timestamp')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training.')
    parser.add_argument('--seq_length', type=int, default=120, help='Sequence length for training.')
    parser.add_argument('--mask_p', type=float, default=0.2, help='Probability for masking.')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden length for BERT4Rec.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch length for training.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs for training.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout probability.')
    parser.add_argument('--weight_decay', type=float, default=0.00, help='Weight decay for optimizer.')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save trained models.')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory to save log files.')
    parser.add_argument('--num_workers', type=int, default=2, help='Dataloader num workers.')

    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    train(
        args.data_name,
        args.data_root,
        args.rating_csv_file,
        args.data_user_col,
        args.data_item_col,
        args.data_chrono_col,
        args.device,
        args.seq_length,
        args.mask_p,
        args.hidden_size,
        args.batch_size,
        args.epochs,
        args.lr,
        args.dropout,
        args.weight_decay,
        args.checkpoint_dir,
        args.log_dir,
        args.num_workers
    )
