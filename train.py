import torch
import pandas as pd
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data import get_datasets_and_loaders
from model import Model
from trainer import Trainer
from tester import Tester


def main():
    parser = argparse.ArgumentParser(description="Train model on Im2LaTeX dataset")
    parser.add_argument("data_directory", type=str,
                        help='directory where {df_train, df_test, df_valid}.pkl files are stored')
    parser.add_argument("image_directory", type=str,
                        help='directory where images are stored')
    parser.add_argument("save_dir", type=str,
                        help="directory to save models")
    parser.add_argument("vocab_path", type=str,
                        help='path to vocab dict (as .pkl file) mapping integer IDs to tokens')
    parser.add_argument("--height", type=int,
                        help="height that all images are scaled to", default=100)
    parser.add_argument("--width", type=int,
                        help="width that all images are scaled to", default=500)
    parser.add_argument("--max_len", type=int,
                        help="maximum number of tokens in target sequences", default=150)
    parser.add_argument("--cuda", action="store_true", default=False,
                        help="use cuda?")
    parser.add_argument("--dropout", type=float,
                        help='dropout probability', default=0.4)
    parser.add_argument("--embedding_size", type=int,
                        help='size of embedding vector in LSTM', default=32)
    parser.add_argument("--batch_size", type=int,
                        help='batch size for training', default=16)
    parser.add_argument("--num_epochs", type=int,
                        help='number of epochs model is trained over', default=23)
    parser.add_argument("--lr", type=float,
                        help='initial learning rate', default=0.0001)
    parser.add_argument("--min_lr", type=float,
                        default=0.00001)
    parser.add_argument('--checkpoint', type=str,
                        help='path to checkpoint if it exists, creates new model if not specified')
    parser.add_argument("--lr_decay", type=float,
                        default=0.5)
    parser.add_argument("--lr_patience", type=int,
                        default=2)
    parser.add_argument("--clip", type=float,
                        help='max norm of gradient', default=2.0)
    parser.add_argument("--test", action="store_true",
                        help="include if testing model from checkpoint", default=False)

    args = parser.parse_args()
    if args.test:
        if args.checkpoint is None:
            print("error, checkpoint not found")
            return
        print("constructing test data...")
        test_dataset, test_dataloader, _ = get_datasets_and_loaders(
            args.data_directory, args.image_directory, "test", args.batch_size,
            args.cuda, args.height, args.width)
        print("loading vocab...")
        vocab = pd.read_pickle(args.vocab_path)
        print("creating model...")
        device = torch.device("cuda" if args.cuda else "cpu")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model = Model(device, len(vocab), args.embedding_size)
        model = model.to(device)
        print("loading weights...")
        model.load_state_dict(checkpoint['model_state_dict'])
        tester = Tester(model, test_dataloader, vocab, args)
        tester.test()
    else:
        print("constructing training data...")
        train_datasets, train_dataloaders, bins = get_datasets_and_loaders(
            args.data_directory, args.image_directory, "train", args.batch_size,
            args.cuda, args.height, args.width)
        print("constructing validation data...")
        val_datasets, val_dataloaders, _ = get_datasets_and_loaders(
            args.data_directory, args.image_directory, "valid", args.batch_size,
            args.cuda, args.height, args.width)
        print("loading vocab...")
        vocab = pd.read_pickle(args.vocab_path)
        print("creating model...")
        device = torch.device("cuda" if args.cuda else "cpu")
        model = Model(device, len(vocab), args.embedding_size, dropout=args.dropout)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        lr_scheduler = ReduceLROnPlateau(optimizer, "min", factor=args.lr_decay,
                                         patience=args.lr_patience,
                                         min_lr=args.min_lr)
        if args.checkpoint is not None:
            print("loading checkpoint...")
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            trainer = Trainer(optimizer, model, lr_scheduler, train_dataloaders,
                              val_dataloaders, args, args.num_epochs, 1)
        else:
            trainer = Trainer(optimizer, model, lr_scheduler, train_dataloaders,
                              val_dataloaders, args, args.num_epochs, 1)
        print("starting training...")
        trainer.train()


if __name__ == "__main__":
    main()
