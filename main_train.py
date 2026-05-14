import torch
import torch.nn as nn
import argparse
import os
import json
import shutil
import numpy as np
from model import *
from dataset_features import codecfake, ASVspoof2019
from CSAM import *
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler, Sampler
import torch.utils.data.sampler as torch_sampler
from collections import defaultdict
from tqdm import tqdm, trange
import random
from utils import *
import eval_metrics as em
import torch.nn.functional as F
import torch.multiprocessing as mp
from balanced_training_utils import (
    create_weighted_sampler, 
    get_loss_function,
    log_class_distribution
)

#mp.set_start_method('spawn', force=True)
torch.set_default_tensor_type(torch.FloatTensor)
torch.multiprocessing.set_start_method('spawn', force=True)

import os
import random
import numpy as np
import torch

def setup_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Per PyTorch più recenti: forza operazioni deterministiche dove possibile
    #torch.use_deterministic_algorithms(True, warn_only=True)

def initParams():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed', type=int, help="random number seed", default=688)
    parser.add_argument("-a", "--access_type", type=str, help="LA or PA", default='LA')
    parser.add_argument("-d", "--path_to_database", type=str, help="dataset path",
                        default='./Codecfake')
    parser.add_argument("-f", "--path_to_features", type=str, help="features path",
                        default='./features/codecfake_xls-r-5')
    parser.add_argument("-f1", "--path_to_features1", type=str, help="cotrain_dataset1_path",
                        default='./features/asvspoof_xls-r-5')
    parser.add_argument("-o", "--out_fold", type=str, help="output folder", 
                        required=False, default='./models/codecfake_model/')

    parser.add_argument("--feat", type=str, help="which feature to use", default='xls-r-5',
                        choices=["mel", "xls-r-5", "xls"])
    parser.add_argument("--feat_len", type=int, help="features length", default=50)
    parser.add_argument('--pad_chop', type=bool, nargs='?', const=True, default=False,
                        help="whether pad_chop in the dataset")
    parser.add_argument('--padding', type=str, default='repeat', choices=['zero', 'repeat', 'silence'],
                        help="how to pad short utterance")

    parser.add_argument('-m', '--model', help='Model arch', default='W2VAASIST',
                        choices=['lcnn','W2VAASIST', 'RawNetLite'])

    parser.add_argument('--train_task', type=str, default='codecfake', 
                        choices=['19LA','codecfake','co-train'], help="training dataset")
    parser.add_argument('--num_epochs', type=int, default=50, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=32, help="Mini batch size for training")
    parser.add_argument('--lr', type=float, default=0.0005, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.5, help="decay learning rate")
    parser.add_argument('--interval', type=int, default=10, help="interval to decay lr")
    parser.add_argument('--beta_1', type=float, default=0.9, help="bata_1 for Adam")
    parser.add_argument('--beta_2', type=float, default=0.999, help="beta_2 for Adam")
    parser.add_argument('--eps', type=float, default=1e-8, help="epsilon for Adam")
    parser.add_argument("--gpu", type=str, help="GPU index", default="0")
    parser.add_argument('--num_workers', type=int, default=8, help="number of workers")
    parser.add_argument('--base_loss', type=str, default="ce", choices=["ce", "bce", "focal"],
                        help="use which loss for basic training (ce=CrossEntropy, focal=FocalLoss)")
    parser.add_argument('--use_weighted_sampler', type= bool, default=True,
                        help="use WeightedRandomSampler for class balancing")
    parser.add_argument("--continue_training", action="store_true", help="Resume training from a checkpoint")
    parser.add_argument("--resume_epoch", type=int, default=None, help="Epoch number to resume from")
    parser.add_argument('--pretrained_model', type=str, default=None,
                    help='Path to pretrained model weights for fine-tuning')

    # Generalized strategy 
    parser.add_argument('--SAM', type=bool, default=False, help="use SAM")
    parser.add_argument('--ASAM', type=bool, default=False, help="use ASAM")
    parser.add_argument('--CSAM', type=bool, default=False, help="use CSAM")
    
    '''# Memory management
    parser.add_argument('--gpu_memory_fraction', type=float, default=0.5,
                        help="Fraction of GPU memory to use")'''

    args = parser.parse_args()


    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    '''# Limita memoria GPU se specificato
    if torch.cuda.is_available() and args.gpu_memory_fraction < 1.0:
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
'''
    # Set seeds
    setup_seed(args.seed)

    if args.continue_training:
        pass
    else:
        # Path for output data
        if not os.path.exists(args.out_fold):
            os.makedirs(args.out_fold)
        else:
            shutil.rmtree(args.out_fold)
            os.mkdir(args.out_fold)

        # Folder for intermediate results
        if not os.path.exists(os.path.join(args.out_fold, 'checkpoint')):
            os.makedirs(os.path.join(args.out_fold, 'checkpoint'))
        else:
            shutil.rmtree(os.path.join(args.out_fold, 'checkpoint'))
            os.mkdir(os.path.join(args.out_fold, 'checkpoint'))

        # Path for input data - verifica che esistano
        if not os.path.exists(args.path_to_features):
            print(f"WARNING: Features path not found: {args.path_to_features}")
            print("Make sure preprocessing is completed first!")

        # Save training arguments
        with open(os.path.join(args.out_fold, 'args.json'), 'w') as file:
            file.write(json.dumps(vars(args), sort_keys=True, separators=('\n', ':')))

        with open(os.path.join(args.out_fold, 'train_loss.log'), 'w') as file:
            file.write("Start recording training loss ...\n")
        with open(os.path.join(args.out_fold, 'dev_loss.log'), 'w') as file:
            file.write("Start recording validation loss ...\n")

    args.cuda = torch.cuda.is_available()
    print('Cuda device available: ', args.cuda)
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args

def adjust_learning_rate(args, lr, optimizer, epoch_num):
    lr = lr * (args.lr_decay ** (epoch_num // args.interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def shuffle(feat, labels):
    shuffle_index = torch.randperm(labels.shape[0])
    feat = feat[shuffle_index]
    labels = labels[shuffle_index]
    return feat, labels

def reshape_features_for_model(features, model_name):
    """Adatta le features al modello specifico"""
    # features arriva come [batch, seq_len, hidden_dim] = [64, 49, 1024]
    if model_name == 'W2VAASIST':
        # W2VAASIST vuole [batch, hidden_dim, seq_len] NON 4D!
        return features.transpose(1, 2)  # [64, 1024, 49]
    else:
        return features

def load_training_checkpoint(checkpoint_path, feat_model, feat_optimizer, device):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

    feat_model.load_state_dict(checkpoint['model_state_dict'])
    feat_model = feat_model.to(device)

    feat_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    early_stopping_counter = checkpoint.get('early_stopping_counter', 0)

    return feat_model, feat_optimizer, start_epoch, best_val_loss, early_stopping_counter

def train(args):
    torch.set_default_tensor_type(torch.FloatTensor)
    # Initialize model - SEZIONE MODIFICATA
    # Caricamento Modelli
    if args.model == 'W2VAASIST':
        feat_model = W2VAASIST().cuda()
    elif args.model == 'RawNetLite':
        feat_model = RawNetLite().cuda()
    elif args.model == 'lcnn':
        feat_model = LCNN().cuda()
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Print model info
    params = sum(p.numel() for p in feat_model.parameters() if p.requires_grad)
    print(f"Model: {args.model}")
    print(f"Trainable parameters: {params:,}")

    # Caricamento ottimizzatore 
    # Setup optimizer
    feat_optimizer = torch.optim.Adam(feat_model.parameters(), lr=args.lr,
                                      betas=(args.beta_1, args.beta_2), eps=args.eps, weight_decay=0.0005)

    if args.SAM or args.CSAM:
        feat_optimizer = torch.optim.Adam
        feat_optimizer = SAM(
            feat_model.parameters(),
            feat_optimizer,
            lr=args.lr,
            betas=(args.beta_1, args.beta_2),
            weight_decay=0.0005
        )

    if args.ASAM:
        feat_optimizer = torch.optim.Adam
        feat_optimizer = SAM(
            feat_model.parameters(),
            feat_optimizer,
            lr=args.lr,
            adaptive=True,
            betas=(args.beta_1, args.beta_2),
            weight_decay=0.0005
        )
    monitor_loss = 'base_loss'
    best_prev_loss = float('inf')
    patience = 10
    counter = 0
    start_epoch = 0

    # Fine-tuning da pretrained net. Carico direttamente la rete con i suoi pesi
    '''if args.pretrained_model is not None:
        print(f"Loading pretrained model from: {args.pretrained_model}")
        feat_model = torch.load(args.pretrained_model, map_location='cpu', weights_only=False).to(args.device)
        feat_optimizer = torch.optim.Adam(
        feat_model.parameters(),
        lr=args.lr,
        betas=(args.beta_1, args.beta_2),
        eps=args.eps,
        weight_decay=0.0005
    )
'''
    # Resume vero
    if args.continue_training:
        if args.resume_epoch is not None:
            checkpoint_path = os.path.join(
                args.out_fold, 'checkpoint', f'checkpoint_epoch_{args.resume_epoch}.pt'
            )
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        else:
            raise ValueError("Please specify --resume_epoch when using --continue_training")

        if os.path.exists(checkpoint_path):
            feat_model, feat_optimizer, start_epoch, best_prev_loss, counter = \
                load_training_checkpoint(checkpoint_path, feat_model, feat_optimizer, args.device)
            print(f"Resumed from: {checkpoint_path}")
            print(f"Start epoch: {start_epoch + 1}")
            print(f"Best val loss so far: {best_prev_loss:.5f}")
            print(f"Early stopping counter: {counter}")
        else:
            print(f"Checkpoint not found: {checkpoint_path}. Starting from scratch.")


    
    
    # Dataset loading based on task
    if args.train_task == 'codecfake':
        print("Loading Codecfake dataset...")
        
        codec_training_set = codecfake(args.access_type, args.path_to_features, 'train',
                                    args.feat, feat_len=args.feat_len, pad_chop=args.pad_chop, padding=args.padding)
        codec_validation_set = codecfake(args.access_type, args.path_to_features, 'dev',
                                    args.feat, feat_len=args.feat_len, pad_chop=args.pad_chop, padding=args.padding)
        
        train_size = len(codec_training_set)
        val_size = len(codec_validation_set)
        print(f"Train samples found: {train_size}")
        print(f"Val samples found: {val_size}")
        
        # NUOVO: Usa WeightedRandomSampler se abilitato
        if args.use_weighted_sampler:
            print("\n Usando WeightedRandomSampler per bilanciamento classi")
            train_sampler = create_weighted_sampler(codec_training_set, verbose=True)
            trainOriDataLoader = DataLoader(
                codec_training_set,
                batch_size=int(args.batch_size),
                shuffle=False,  # IMPORTANTE: False quando si usa sampler!
                num_workers=args.num_workers,
                sampler=train_sampler,
                pin_memory=args.cuda,
                drop_last=True
            )
        else:
            # Fallback al metodo originale
            if train_size < 100000:
                print(f"Using shuffle=True since only {train_size} files found")
                trainOriDataLoader = DataLoader(
                    codec_training_set,
                    batch_size=int(args.batch_size),
                    shuffle=True,
                    num_workers=args.num_workers,
                    pin_memory=args.cuda,
                    drop_last=True
                )
            else:
                trainOriDataLoader = DataLoader(
                    codec_training_set,
                    batch_size=int(args.batch_size),
                    shuffle=False,
                    num_workers=args.num_workers,
                    persistent_workers=True if args.num_workers > 0 else False,
                    pin_memory=args.cuda,
                    sampler=torch_sampler.SubsetRandomSampler(range(train_size))
                )
        
        if val_size < 40000:
            print(f"Using shuffle=False for val since only {val_size} files found (expected 40000)")
            valOriDataLoader = DataLoader(codec_validation_set, batch_size=int(args.batch_size),
                                        shuffle=False, num_workers=args.num_workers,
                                        pin_memory=args.cuda)
        else:
            valOriDataLoader = DataLoader(codec_validation_set, batch_size=int(args.batch_size),
                                        shuffle=False, num_workers=args.num_workers,
                                        persistent_workers=True if args.num_workers > 0 else False,
                                        sampler=torch_sampler.SubsetRandomSampler(range(val_size)))

    elif args.train_task == '19LA':
        print("Loading ASVspoof2019 dataset...")
        asv_training_set = ASVspoof2019(args.access_type, args.path_to_features1, 'train',
                                    args.feat, feat_len=args.feat_len, pad_chop=args.pad_chop, padding=args.padding)
        asv_validation_set = ASVspoof2019(args.access_type, args.path_to_features1, 'dev',
                                    args.feat, feat_len=args.feat_len, pad_chop=args.pad_chop, padding=args.padding)
        
        # NUOVO: Usa WeightedRandomSampler se abilitato
        if args.use_weighted_sampler:
            print("\n Usando WeightedRandomSampler per bilanciamento classi")
            train_sampler = create_weighted_sampler(asv_training_set, verbose=True)
            trainOriDataLoader = DataLoader(
                asv_training_set,
                batch_size=int(args.batch_size),
                shuffle=False,
                num_workers=args.num_workers,
                sampler=train_sampler
            )
        else:
            trainOriDataLoader = DataLoader(
                asv_training_set,
                batch_size=int(args.batch_size),
                shuffle=False,
                num_workers=args.num_workers,
                sampler=torch_sampler.SubsetRandomSampler(range(len(asv_training_set)))
            )
        valOriDataLoader = DataLoader(asv_validation_set, batch_size=int(args.batch_size),
                                      shuffle=False, num_workers=args.num_workers,
                                      sampler=torch_sampler.SubsetRandomSampler(range(len(asv_validation_set))))

    elif args.train_task == 'co-train':
        print("Loading co-training datasets...")
        # Load both datasets
        asv_training_set = ASVspoof2019(args.access_type, args.path_to_features1, 'train',
                                    args.feat, feat_len=args.feat_len, pad_chop=args.pad_chop, padding=args.padding)
        asv_validation_set = ASVspoof2019(args.access_type, args.path_to_features1, 'dev',
                                      args.feat, feat_len=args.feat_len, pad_chop=args.pad_chop, padding=args.padding)
        codec_training_set = codecfake(args.access_type, args.path_to_features, 'train',
                                    args.feat, feat_len=args.feat_len, pad_chop=args.pad_chop, padding=args.padding)
        codec_validation_set = codecfake(args.access_type, args.path_to_features, 'dev',
                                      args.feat, feat_len=args.feat_len, pad_chop=args.pad_chop, padding=args.padding)

        # Concat datasets
        training_set = ConcatDataset([codec_training_set, asv_training_set])
        validation_set = ConcatDataset([codec_validation_set, asv_validation_set])

        train_total_samples_codec = len(codec_training_set)
        train_total_samples_asv = len(asv_training_set)
        train_total_samples_combined = len(training_set)
        train_codec_weight = train_total_samples_codec / train_total_samples_combined
        train_asv_weight = train_total_samples_asv / train_total_samples_combined

        if args.CSAM:
            print("\n Co-training con CSAMSampler")
            trainOriDataLoader = DataLoader(training_set, batch_size=int(args.batch_size),
                                          shuffle=False, num_workers=args.num_workers,
                                          sampler=CSAMSampler(dataset=training_set,
                                                             batch_size=int(args.batch_size),
                                                             ratio_dataset1=train_codec_weight,
                                                             ratio_dataset2=train_asv_weight))
        else:
            # NUOVO: Usa WeightedRandomSampler per co-training
            if args.use_weighted_sampler:
                print("\n Co-training con WeightedRandomSampler")
                train_sampler = create_weighted_sampler(training_set, verbose=True)
                trainOriDataLoader = DataLoader(
                    training_set,
                    batch_size=int(args.batch_size),
                    shuffle=False,
                    num_workers=args.num_workers,
                    sampler=train_sampler,
                    pin_memory=args.cuda
                )
            else:
                trainOriDataLoader = DataLoader(
                    training_set,
                    batch_size=int(args.batch_size),
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=args.cuda,
                    sampler=torch_sampler.SubsetRandomSampler(range(len(training_set)))
                )
        
        valOriDataLoader = DataLoader(validation_set, batch_size=int(args.batch_size),
                                    shuffle=False, num_workers=args.num_workers,
                                    sampler=torch_sampler.SubsetRandomSampler(range(len(validation_set))))


    '''trainOri_flow = iter(trainOriDataLoader)
    valOri_flow = iter(valOriDataLoader)'''

    # Loss setup con supporto per Focal Loss
    print(f"\n{'='*80}")
    criterion = get_loss_function(
        loss_type=args.base_loss,
        model_name=args.model,
        device=args.device,
        alpha_focal=None  # Usa default, oppure specifica [0.25, 0.75]
    )
    print(f"{'='*80}\n")
 

    # Training loop
    for epoch_num in tqdm(range(start_epoch, args.num_epochs), desc="Epochs"):
        
        feat_model.train()
        trainlossDict = defaultdict(list)
        devlossDict = defaultdict(list)
        
        adjust_learning_rate(args, args.lr, feat_optimizer, epoch_num)

        # Training phase
        '''for i in trange(len(trainOriDataLoader), desc=f"Training Epoch {epoch_num+1}"):
            try:
                featOri, audio_fnOri, labelsOri = next(trainOri_flow)
            except StopIteration:
                trainOri_flow = iter(trainOriDataLoader)
                featOri, audio_fnOri, labelsOri = next(trainOri_flow)'''

        for i, (featOri, audio_fnOri, labelsOri) in enumerate(tqdm(trainOriDataLoader, desc=f"Training Epoch {epoch_num+1}")):
        
            feat = reshape_features_for_model(featOri, args.model).to(args.device)
            labels = labelsOri.to(args.device)

            if args.SAM or args.ASAM or args.CSAM:
                enable_running_stats(feat_model)
                feats, feat_outputs = feat_model(feat)
                feat_loss = criterion(feat_outputs, labels)
                feat_loss.mean().backward()
                feat_optimizer.first_step(zero_grad=True)

                disable_running_stats(feat_model)
                feats, feat_outputs = feat_model(feat)
                criterion(feat_outputs, labels).mean().backward()
                feat_optimizer.second_step(zero_grad=True)
            else:
                feat_optimizer.zero_grad()
                feats, feat_outputs = feat_model(feat)
                feat_loss = criterion(feat_outputs, labels)
                feat_loss.backward()
                feat_optimizer.step()

            trainlossDict['base_loss'].append(feat_loss.item())
            
            # Monitoring bilanciamento ogni 200 batch
            if i % 200 == 0 and args.use_weighted_sampler:
                log_class_distribution(
                    epoch=epoch_num + 1,
                    phase='train',
                    outputs=feat_outputs.detach(),
                    labels=labels.detach()
                )

            # Periodic memory cleanup
            if i % 50 == 0:
                torch.cuda.empty_cache()

        # Validation phase
        feat_model.eval()
        with torch.no_grad():
            #ip1_loader, idx_loader, score_loader = [], [], []
            
            '''for i in trange(len(valOriDataLoader), desc="Validation"):
                try:
                    featOri, audio_fnOri, labelsOri = next(valOri_flow)
                except StopIteration:
                    valOri_flow = iter(valOriDataLoader)
                    featOri, audio_fnOri, labelsOri = next(valOri_flow)'''

            for i, (featOri, audio_fnOri, labelsOri) in enumerate(tqdm(valOriDataLoader, desc="Validation")):
                
                feat = reshape_features_for_model(featOri, args.model).to(args.device)
                labels = labelsOri.to(args.device)
                feats, feat_outputs = feat_model(feat)

                if args.base_loss == "bce":
                    feat_loss = criterion(feat_outputs, labels.unsqueeze(1).float())
                    score = feat_outputs[:, 0]
                else:
                    feat_loss = criterion(feat_outputs, labels)
                    score = F.softmax(feat_outputs, dim=1)[:, 0]

                #ip1_loader.append(feats)
                #idx_loader.append(labels)
                devlossDict["base_loss"].append(feat_loss.item())
                #score_loader.append(score)

        # Calculate metrics
        valLoss = np.nanmean(devlossDict[monitor_loss])
        train_loss = np.nanmean(trainlossDict[monitor_loss])
        
        print(f"\nEpoch {epoch_num+1}/{args.num_epochs}")
        print(f"Train Loss: {train_loss:.5f}")
        print(f"Val Loss: {valLoss:.5f}")
        
        # Save logs
        with open(os.path.join(args.out_fold, "train_loss.log"), "a") as log:
            log.write(f"{epoch_num}\t{train_loss:.5f}\n")
        with open(os.path.join(args.out_fold, "dev_loss.log"), "a") as log:
            log.write(f"{epoch_num}\t{valLoss:.5f}\n")


        # Save best model
        if valLoss < best_prev_loss:
            torch.save(feat_model, os.path.join(args.out_fold, 'anti-spoofing_feat_model.pt'))
            best_prev_loss = valLoss
            counter = 0
            print(f"Best model saved with val loss: {valLoss:.5f}")
        else:
            counter += 1
            print(f"No improvement for {counter} epoch(s)")

        

        # Save checkpoints
        checkpoint = {
            'epoch': epoch_num,
            'model_state_dict': feat_model.state_dict(),
            'optimizer_state_dict': feat_optimizer.state_dict(),
            'best_val_loss': best_prev_loss,
            'early_stopping_counter': counter,
            'train_loss': train_loss,
            'val_loss': valLoss,
            'args': vars(args),
        }
        epoch_ckpt_path = os.path.join(
            args.out_fold, 'checkpoint', f'checkpoint_epoch_{epoch_num+1}.pt'
        )
        torch.save(checkpoint, epoch_ckpt_path)
        # Clear cache
        torch.cuda.empty_cache()

        if counter >= patience:
            print("Early stopping triggered")
            break

    return feat_model

if __name__ == "__main__":
    args = initParams()
    model = train(args)
