import argparse
import datetime
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import build_fdb_data, collate_fn
from models import build_models, make_criterion
from engine import Engine


def get_args_parser():
    parser = argparse.ArgumentParser("DETR for Argument Mining")

    # Optimization parameters
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=2, type=int, help="Batch Size")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay regularization factor")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="Start epoch")
    parser.add_argument("--epochs", default=300, type=int, help="Number of trainig epochs")
    parser.add_argument("--lr_drop", default=200, type=int, help="Drop learning rate each lr_drop epochs")
    parser.add_argument("--clip_max_norm", default=0.1, type=float, help="Gradient clipping max norm")
    parser.add_argument("--train_trans_from_epoch", default=0, type=int, help="train the transformer module from the specified epoch (-1 to disable)")
    parser.add_argument("--transformer_lr", default=1e-5, type=float, help="learning rate for the transformer")

    # Model parameters
    parser.add_argument("--hidden_dim", default=1024, type=int, help="MLP hidden dimension")
    parser.add_argument("--num_queries", default=40, type=int, help="Number of query slots")
    parser.add_argument("--class_depth", default=1, type=int, help="Layers in the classification head")
    parser.add_argument("--bbox_depth", default=3, type=int, help="Layers in the bbox regression head")
    parser.add_argument("--frozen_weights", type=str, default=None, help="Path to the pretrained model")
    parser.add_argument("--resume", type=str, default=None, help="resume from checkpoint")
    parser.add_argument("--init_last_biases", default=True, help="Init last layer biases using logits distribution")
    parser.add_argument("--dropout", default=0, help="Dropout value applied after each linear layers (0 to disable) ")
    parser.add_argument("--init_weight", default=None, help="Initialization of the MLP weights")
    parser.add_argument("--pretrained", type=bool, default=True, help="For choosing a not pretrained model")
    parser.add_argument("--glob_attn_words", type=bool, default=False, help="Global attentuon words assigned to highly correlated words")


    # Loss coefficients
    parser.add_argument("--losses", default=["labels", "boxes", "cardinality"], nargs="+", help="List of losses to compute, chose between: labels, boxes, cardinality, overlap")
    parser.add_argument("--ce_loss_coef", default=1, type=float, help="CrossEntropy coefficient in the loss")
    parser.add_argument("--block_ce_for", default=-1, type=float, help="Block CrossEntropy loss for n epochs")
    parser.add_argument("--bbox_loss_coef", default=1, type=float, help="L1 box coefficient in the loss")
    parser.add_argument("--giou_loss_coef", default=0.5, type=float, help="giou box coefficient in the loss")
    parser.add_argument("--overlap_loss_coef", default=0.5, type=float, help="Overlap box coefficient in the loss")
    parser.add_argument("--focal_loss_gamma", default=2, type=float, help="Focal Loss parameter (0 to disable)")
    parser.add_argument("--no_class_weight", default=False, action='store_true', help="Don't use class weights")
    parser.add_argument("--effective_num", default=False, action='store_true', help="For effective number of object weights")
    parser.add_argument("--beta", default=0, type=float, help="beta parameter for effective number of object weights")

    # Dataset parameters
    parser.add_argument("--input_path", default="./input/feedback-prize-2021/", type=str, help="Folder where the inputs are")
    parser.add_argument("--val_size", default=0.1, type=float, help="Size of the validation set in the range (0, 1)")
    parser.add_argument("--test_size", default=0.15, type=float, help="Size of the test set in the range (0, 1)")
    parser.add_argument("--no_preprocessing", default=False, action='store_true', help="Don't apply preprocessing to the dataset")
    parser.add_argument("--num_workers", default=2, type=int, help="Workers used by the DataLoader")
    parser.add_argument("--dataset_size", default=1.0, type=float, help="[0, 1], 1 for full dataset")
    parser.add_argument("--no_align_target", default=False, action='store_true', help="Don't se aligned target")

    # Other parameters
    parser.add_argument("--output_dir", default="./outputs", type=str, help="Folder where the outputs will be saved")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=42, type=int, help="seed for reproducibility")
    parser.add_argument("--eval", default=False, action='store_true', help="only evaluate the validation set and exit")
    parser.add_argument("--test", default=False, action='store_true', help="only evaluate the test set and exit")

    return parser


def main(args):
    vargs = vars(args)
    pad = max(len(k) for k in vargs.keys())
    print("ARGUMENTS".rjust(pad), "-", "VALUES")
    for key in sorted(vargs.keys()):
        print(key.rjust(pad), ":", vargs[key])

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print()
    print("Loading Dataset...")

    dataset_train, dataset_val, dataset_test, postprocessor, num_classes, freqs = build_fdb_data(args)
    print('Using frequencies:', freqs)

    print("Dataset loaded")
    print()
    print("Loading Models...")

    ce_loss_coef = args.ce_loss_coef
    if args.block_ce_for > 0: 
        args.ce_loss_coef = 0 # block ce

    
    tokenizer, model, criterion = build_models(num_classes, freqs, args)
    model.to(device)

    model.set_transformer_trainable(True)
    model.set_backbone_trainable(False)
    
    attention_words =[]
    if args.glob_attn_words:
        attention_words = [1219, 6427, 5848, 41752, 766, 679, 959]

    print("Models Loaded")
    print()

    optimizer = torch.optim.AdamW(
        model.last_layers_parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    data_loader_train = DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    data_loader_val = DataLoader(
        dataset_val,
        shuffle=False,
        batch_size=args.batch_size,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    data_loader_test = DataLoader(
        dataset_test,
        shuffle=False,
        batch_size=args.batch_size,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location="cpu")
        model.load_state_dict(checkpoint["model"])

    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        if (
            not args.eval
            and "optimizer" in checkpoint
            and "lr_scheduler" in checkpoint
            and "epoch" in checkpoint
        ):
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1

    engine = Engine()

    if args.eval:
        postprocessor.reset_results()
        report = engine.evaluate(
            tokenizer=tokenizer,
            model=model,
            criterion=criterion,
            postprocessor=postprocessor,
            data_loader=data_loader_val,
            epoch=0,
            device=device,
            attention_words=attention_words
        )

        print(report.to_string())
        return postprocessor.results

    if args.test:
        postprocessor.reset_results()
        report = engine.evaluate(
            tokenizer=tokenizer,
            model=model,
            criterion=criterion,
            postprocessor=postprocessor,
            data_loader=data_loader_test,
            epoch=0,
            device=device,
            attention_words=attention_words
        )

        print(report.to_string())
        return postprocessor.results

    output_dir = engine.set_outputs(args.output_dir)
    print("Start training")
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)
    print('- '*50)
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if epoch == args.train_trans_from_epoch:
            print('Start training Transformer')
            model.set_transformer_trainable(True)
            optimizer.add_param_group({'params': model.transformer_parameters(), 'lr': args.transformer_lr})

            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print("number of params:", n_parameters)
            print('- '*50)
        
        if epoch == args.block_ce_for: # restore ce_loss_coef
            args.ce_loss_coef = ce_loss_coef
            criterion = make_criterion(num_classes, freqs, args, device)

        postprocessor.reset_results()
        report = engine.train_one_epoch(
            tokenizer=tokenizer,
            model=model,
            criterion=criterion,
            postprocessor=postprocessor,
            data_loader=data_loader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            max_norm=args.clip_max_norm,
            attention_words=attention_words
        )
        print(report.to_string())
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / "checkpoint.pth"]
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f"checkpoint{epoch:04}.pth")
            for checkpoint_path in checkpoint_paths:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "args": args,
                    },
                    checkpoint_path,
                )

        postprocessor.reset_results()
        report = engine.evaluate(
            tokenizer=tokenizer,
            model=model,
            criterion=criterion,
            postprocessor=postprocessor,
            data_loader=data_loader_val,
            epoch=epoch,
            device=device,
            attention_words=attention_words
        )

        print(report.to_string())
        print('- '*50)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    return postprocessor.results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "DETR training and evaluation script",
        parents=[get_args_parser()],
        add_help=False,
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
