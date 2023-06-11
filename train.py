import argparse
import os

import torch
from dotenv import load_dotenv

from src.recsys.helpers import get_handler, get_dataloaders
from src.recsys.models.sequential import BERT4Rec


def test(
        data_name,
        data_root,
        data_user_col,
        data_item_col,
        data_chrono_col,
        device,
        seq_length,
        batch_size,
        checkpoint_dir,
        log_dir,
        num_workers=10
):
    test_loader = get_dataloaders(
        'test',
        data_name,
        data_root,
        data_user_col,
        data_item_col,
        data_chrono_col,
        seq_length,
        batch_size,
        num_workers
    )

    checkpoint_prefix = f"bert4rec_{data_name}"
    model = BERT4Rec.load_from_checkpoint(
        os.path.join(checkpoint_dir, checkpoint_prefix + "_best.ckpt"),
        vocab_size=test_loader.dataset.vocab_size,
        mask_token=test_loader.dataset.mask_token,
        pad_token=test_loader.dataset.pad_token,
        map_location=device
    )
    handler = get_handler(
        epochs=0,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        checkpoint_prefix=checkpoint_prefix,
        device=device
    )
    handler.test(model, test_loader)


def train(
        data_name,
        data_root,
        data_user_col,
        data_item_col,
        data_chrono_col,
        device,
        seq_length,
        mask_p,
        pretrained,
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
    train_loader, val_loader, test_loader = get_dataloaders(
        ['train', 'val', 'test'],
        data_name,
        data_root,
        data_user_col,
        data_item_col,
        data_chrono_col,
        seq_length,
        batch_size,
        num_workers,
        mask_p
    )
    if pretrained:
        model = BERT4Rec.load_from_checkpoint(
            os.path.join(checkpoint_dir, pretrained),
            vocab_size=train_loader.dataset.vocab_size,
            mask_token=train_loader.dataset.mask_token,
            pad_token=train_loader.dataset.pad_token,
            map_location=device
        )
    else:
        model = BERT4Rec(
            seq_length=seq_length,
            vocab_size=train_loader.dataset.vocab_size,
            mask_token=train_loader.dataset.mask_token,
            pad_token=train_loader.dataset.pad_token,
            hidden_size=hidden_size,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay
        )
    checkpoint_prefix = f"{model.__class__.__name__.lower()}_{data_name}"
    handler = get_handler(
        epochs=epochs,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        checkpoint_prefix=checkpoint_prefix,
        device=device
    )
    handler.fit(model, train_loader, val_loader)
    model = BERT4Rec.load_from_checkpoint(
        os.path.join(checkpoint_dir, checkpoint_prefix + "_best.ckpt"),
        vocab_size=test_loader.dataset.vocab_size,
        mask_token=test_loader.dataset.mask_token,
        pad_token=test_loader.dataset.pad_token,
        map_location=device
    )
    handler.test(model, test_loader)


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description='Training parameters for BERT4Rec.')
    parser.add_argument('--data_name', type=str, default=os.getenv("MOVIELENS_VERSION"))
    parser.add_argument('--data_root', type=str, default=os.getenv("DATABASE_ROOT"))
    parser.add_argument('--data_user_col', type=str, default='userId')
    parser.add_argument('--data_item_col', type=str, default='movieId')
    parser.add_argument('--data_chrono_col', type=str, default='timestamp')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training.')
    parser.add_argument('--seq_length', type=int, default=120, help='Sequence length for training.')
    parser.add_argument('--mask_p', type=float, default=0.2, help='Probability for masking.')
    parser.add_argument('--pretrained', type=str, default=None, help='Path to the pretrained checkpoint')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden length for BERT4Rec.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch length for training.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs for training.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout probability.')
    parser.add_argument('--weight_decay', type=float, default=0.00, help='Weight decay for optimizer.')
    parser.add_argument('--checkpoint_dir', type=str, default='./resources/checkpoints',
                        help='Directory to save trained models.')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory to save log files.')
    parser.add_argument('--num_workers', type=int, default=2, help='Dataloader num workers.')

    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    if args.epochs == 0:
        test(
            args.data_name,
            args.data_root,
            args.data_user_col,
            args.data_item_col,
            args.data_chrono_col,
            args.device,
            args.seq_length,
            args.batch_size,
            args.checkpoint_dir,
            args.log_dir,
            args.num_workers
        )
        exit(0)

    train(
        args.data_name,
        args.data_root,
        args.data_user_col,
        args.data_item_col,
        args.data_chrono_col,
        args.device,
        args.seq_length,
        args.mask_p,
        args.pretrained,
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
