###############################################################################
# Language Modeling on Wikitext-2
#
# This file generates new sentences sampled from the language model. 
# 
# The code is adapted from https://github.com/pytorch/examples/tree/main/word_language_model
#
###############################################################################
import argparse
import torch

import data
from model import LanguageLSTM

def generate_text(device, checkpoint, data_source, words, temperature, log_interval):
    
    with open(checkpoint, 'rb') as f:
        model = torch.load(f, map_location=device)
    model.eval()

    corpus = data.Corpus(data_source)
    ntokens = len(corpus.dictionary)

    hidden = model.init_hidden(1)
    input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)

    generated_text = []
    with torch.no_grad():  # no tracking history
        for i in range(words):
            output, hidden = model(input, hidden)
            word_weights = output.squeeze().div(temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            input.fill_(word_idx)

            word = corpus.dictionary.idx2word[word_idx]
            generated_text.append(word)

            if i % log_interval == 0:
                print('| Generated {}/{} words'.format(i, words))
    
    return generated_text

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')
    parser.add_argument('--data', type=str, default='./wikitext-2',
                        help='location of the data corpus')
    parser.add_argument('--checkpoint', type=str, default='./model.pt',
                        help='model checkpoint to use')
    parser.add_argument('--outf', type=str, default='generated.txt',
                        help='output file for generated text')
    parser.add_argument('--words', type=int, default='1000',
                        help='number of words to generate')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disable CUDA')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disable MPS')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='temperature - higher will increase diversity')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='reporting interval')
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

    if args.temperature < 1e-3:
        parser.error("--temperature has to be greater or equal 1e-3.")
    
    generated_text = generate_text(device, args.checkpoint, args.data, args.words, args.temperature, args.log_interval)

    with open(args.outf, 'w') as outf:
        for i, word in enumerate(generated_text):
            outf.write(word + ('\n' if i % 20 == 19 else ' '))
        print('Generated text saved to', args.outf)
    