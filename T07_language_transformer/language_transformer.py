import torch
import torch.nn as nn
import argparse
import math
from matplotlib import pyplot as plt
import numpy as np
import time
import data
from model import TransformerModel

def batchify(device, data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

def get_batch(source, i):
    seq_len = min(35, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def evaluate(device, model, eval_data, criterion, seq_len, ntokens):
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, seq_len):
            data, targets = get_batch(eval_data, i)
            data, targets = data.to(device), targets.to(device)
            output = model(data)
            output = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(eval_data) - 1)

def train(device, model, epoch, train_data, criterion, lr, log_interval, seq_len, ntokens):
    model.train()
    total_loss = 0.
    loss_all = []
    data_cnt = []
    start_time = time.time()
    for batch, i in enumerate(range(0, train_data.size(0) - 1, seq_len)):
        data, targets = get_batch(train_data, i)
        data, targets = data.to(device), targets.to(device)
        model.zero_grad()
        output = model(data)
        output = output.view(-1, ntokens)
        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-lr)
        total_loss += loss.item()
        loss_all.append(loss.item())
        data_cnt.append(len(data))
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // seq_len, lr,
                elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
    return np.average(loss_all, weights=data_cnt)

def language_model(device, data_source="../data/wikitext-2", epochs=50, batch_size=20, eval_batch_size=10, lr=20, emsize=200, nhead=2, nhid=200, nlayers=2, seq_len=35, dropout=0.2, tied=False, save='model.pt', plot=False):

    corpus = data.Corpus(data_source)
    ntokens = len(corpus.dictionary)
    train_data = batchify(device, corpus.train, batch_size)
    val_data = batchify(device, corpus.valid, eval_batch_size)
    test_data = batchify(device, corpus.test, eval_batch_size)
    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
    criterion = nn.NLLLoss()
    best_val_loss = None
    train_losses = []
    val_losses = []
    try:
        for epoch in range(1, epochs+1):
            epoch_start_time = time.time()
            train_loss_tmp = train(device, model, epoch, train_data, criterion, lr, 200, seq_len, ntokens)
            val_loss_tmp = evaluate(device, model, val_data, criterion, seq_len, ntokens)
            train_losses.append(train_loss_tmp)
            val_losses.append(val_loss_tmp)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                             val_loss_tmp, math.exp(val_loss_tmp)))
            print('-' * 89)
            if not best_val_loss or val_loss_tmp < best_val_loss:
                with open(save, 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss_tmp
            else:
                lr /= 4.0
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
    test_loss = evaluate(device, model, test_data, criterion, seq_len, ntokens)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)

    if plot:
        plt.plot(train_losses, label='train')
        plt.plot(val_losses, label='val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('train_val_loss.png')
        plt.close()

    return train_losses, val_losses

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Transformer Language Model')
    parser.add_argument('--data', type=str, default='../data/wikitext-2',
                        help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--eval_batch_size', type=int, default=10, metavar='N',
                        help='eval batch size')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='upper epoch limit')
    parser.add_argument('--emsize', type=int, default=200, metavar='N',
                        help='size of word embeddings')
    parser.add_argument('--nhead', type=int, default=2, metavar='N',
                        help='number of heads in Transformer')
    parser.add_argument('--nhid', type=int, default=200, metavar='N',
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2, metavar='N',
                        help='number of layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--lr', type=float, default=5,
                        help='initial learning rate')
    parser.add_argument('--tied', action='store_true',
                        help='tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disable CUDA')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disable MPS')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='model.pt',
                        help='path to save the final model')
    parser.add_argument('--seq-len', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--plot', action='store_true', default=False,
                        help='plot the results')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()
    device = torch.device("cpu")
    if use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif use_mps:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    train_losses, val_losses = language_model(device, args.data, args.epochs, args.batch_size, args.eval_batch_size, args.lr, args.emsize, args.nhead, args.nhid, args.nlayers, args.seq_len, args.dropout, args.tied, args.save, args.plot)
