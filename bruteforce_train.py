# encoding: utf-8
from __future__ import unicode_literals, print_function

import gc
import json
import os
import glob
import io
import itertools
import numpy as np
import random
from time import time
import torch
from torch import nn
import pickle
import shutil
import math
import six
from torch.autograd import Variable

import evaluator
from backup_models import Transformer
from exp_moving_avg import ExponentialMovingAverage
import optimizer as optim
from torchtext import data
import utils
from config import get_train_args
from fp16_utils import FP16_Optimizer, FP16_Module
import preprocess
from data_parallel import data_parallel as dp

def init_weights(m):
    if type(m) == nn.Linear:
        input_dim = m.weight.size(1)
        # LeCun Initialization
        m.weight.data.uniform_(-math.sqrt(3.0 / input_dim),
                                math.sqrt(3.0 / input_dim))
        # My custom initialization
        # m.weight.data.uniform_(-3. / input_dim, 3. / input_dim)

        # Xavier Initialization
        # output_dim = m.weight.size(0)
        # m.weight.data.uniform_(-math.sqrt(6.0 / (input_dim + output_dim)),
        #                         math.sqrt(6.0 / (input_dim + output_dim)))

        if m.bias is not None:
            m.bias.data.fill_(0.)


def save_checkpoint(state, is_best, model_path_, best_model_path_):
    torch.save(state, model_path_)
    if is_best:
        shutil.copyfile(model_path_, best_model_path_)


global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count, sofar):
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new[0]) + 2)
    max_tgt_in_batch = max(max_tgt_in_batch, len(new[1]) + 1)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


def save_output(hypotheses, vocab, outf):
    # Save the Hypothesis to output file
    with io.open(outf, 'w') as fp:
        for sent in hypotheses:
            words = [vocab[y] for y in sent]
            fp.write(' '.join(words) + '\n')


def tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' in name:
            dec += param.nelement()
    print('encoder: ', enc)
    print('decoder: ', dec)


def report_func(epoch, batch, num_batches, start_time, report_stats,
                report_every):
    """
    This is the user-defined batch-level training progress
    report function.
    Args:
        epoch(int): current epoch count.
        batch(int): current batch count.
        num_batches(int): total number of batches.
        start_time(float): last report time.
        lr(float): current learning rate.
        report_stats(Statistics): old Statistics instance.
    Returns:
        report_stats(Statistics): updated Statistics instance.
    """
    if batch % report_every == -1 % report_every:
        report_stats.output(epoch, batch + 1, num_batches, start_time)
        report_stats = utils.Statistics()

    return report_stats


class CalculateBleu(object):
    def __init__(self, model, test_data, key, batch=50, max_decode_len=50,
                 beam_size=1, alpha=0.6, max_sent=None):
        self.model = model
        self.test_data = test_data
        self.key = key
        self.batch = batch
        self.device = -1
        self.max_decode_length = max_decode_len
        self.beam_size = beam_size
        self.alpha = alpha
        self.max_sent = max_sent

    def __call__(self):
        self.model.eval()
        references = []
        hypotheses = []
        for i in range(0, len(self.test_data), self.batch):
            sources, targets = zip(*self.test_data[i:i + self.batch])
            references.extend(t.tolist() for t in targets)
            x_block = utils.source_pad_concat_convert(sources,
                                                      device=None)
            x_block = Variable(torch.LongTensor(x_block).type(utils.LONG_TYPE),
                               requires_grad=False)
            ys = self.model.translate(x_block,
                                      self.max_decode_length,
                                      beam=self.beam_size,
                                      alpha=self.alpha)
            hypotheses.extend(ys)
            if self.max_sent is not None and \
                    ((i + 1) > self.max_sent):
                break

            # Log Progress
            if self.max_sent is not None:
                den = self.max_sent
            else:
                den = len(self.test_data)
            print("> Completed: [ %d / %d ]" % (i, den), end='\r')

        bleu = evaluator.BLEUEvaluator().evaluate(references, hypotheses)
        print('BLEU:', bleu.score_str())
        print('')
        return bleu.bleu, hypotheses


def main():
    best_score = 0
    args = get_train_args()
    print(json.dumps(args.__dict__, indent=4))

    # Set seed value
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.gpu:
        torch.cuda.manual_seed_all(args.seed)

    # Reading the int indexed text dataset
    # train_data = np.load(os.path.join(args.input, args.data + ".train.npy"), allow_pickle=True)
    # train_data = train_data.tolist()
    # dev_data = np.load(os.path.join(args.input, args.data + ".valid.npy"), allow_pickle=True)
    # dev_data = dev_data.tolist()
    # test_data = np.load(os.path.join(args.input, args.data + ".test.npy"), allow_pickle=True)
    # test_data = test_data.tolist()

    # Reading the vocab file
    with open(os.path.join(args.input, args.data + '.vocab.pickle'), 'rb') as f:
        id2w = pickle.load(f)

    w2id = {word: index for index, word in id2w.items()}
    
    # load the required pruned vocab here
    #vocab = [i    for i in id2w]
    good_words, vocab = torch.load(os.path.join(args.input, 'good_words_in_vocab.pt'))

    source_path = args.src
    target_path = args.pred
    args.tok = False
    args.max_seq_len = 70

    # Train Dataset
    source_data = preprocess.make_dataset(source_path, w2id, args.tok)
    target_data = preprocess.make_dataset(target_path, w2id, args.tok)
    assert len(source_data) == len(target_data)
    train_data = [(s, t) for s, t in six.moves.zip(source_data, target_data)
                  if 0 < len(s) < args.max_seq_len
                  and 0 < len(t) < args.max_seq_len]

    print('Size of data for attack:', len(train_data))
    args.id2w = id2w
    args.n_vocab = len(id2w)

    # Define Model
    log_dir = os.path.abspath(os.path.join(args.input,'..'))
    log_file = open(glob.glob(os.path.join(log_dir,"*.log"))[0],"a")
    print("inside bruteforce context and topk:",(args.context,args.topk),file=log_file)
    log_file.close()
    model = eval(args.model)(args)
    model.apply(init_weights)
    
    tally_parameters(model)
    if args.gpu >= 0:
        model.cuda(args.gpu)
    print(model)

    #optimizer = optim.TransformerAdamTrainer(model, args)
    ema = ExponentialMovingAverage(decay=0.999)
    ema.register(model.state_dict())

    '''if args.fp16:
        model = FP16_Module(model)
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=args.static_loss_scale,
                                   dynamic_loss_scale=args.dynamic_loss_scale,
                                   dynamic_loss_args={'init_scale': 2 ** 16},
                                   verbose=False)'''

    checkpoint = torch.load(args.best_model_file)
    print("=> loaded checkpoint '{}' (epoch {}, best score {})".
          format(args.best_model_file,
                 checkpoint['epoch'],
                 checkpoint['best_score']))

    state_dict = checkpoint['state_dict']
    if args.label_smoothing == 0:
        weight = torch.ones(args.n_vocab)
        weight[model.padding_idx] = 0
        state_dict['criterion.weight'] = weight
        state_dict.pop('one_hot')
    model.load_state_dict(state_dict)

    # put the model in train mode. But change the requires_grad var to False
    # for all parameters except weights
    _flag = True
    for param in model.parameters():
        param.requires_grad = _flag
        _flag = False
    model.eval()

    # optionally resume from a checkpoint
    # if args.resume:
    #     if os.path.isfile(args.model_file):
    #         print("=> loading checkpoint '{}'".format(args.model_file))
    #         checkpoint = torch.load(args.model_file)
    #         args.start_epoch = checkpoint['epoch']
    #         best_score = checkpoint['best_score']
    #         model.load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         print("=> loaded checkpoint '{}' (epoch {})".
    #               format(args.model_file, checkpoint['epoch']))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args.model_file))

    # train_data = dev_data
    src_data, trg_data = list(zip(*train_data))
    total_src_words = len(list(itertools.chain.from_iterable(src_data)))
    total_trg_words = len(list(itertools.chain.from_iterable(trg_data)))
    iter_per_epoch = (total_src_words + total_trg_words) // (2 * args.wbatchsize)
    print('Approximate number of iter/epoch =', iter_per_epoch)
    time_s = time()

    num_grad_steps = 0
    n_vocab = len(vocab)
    model.weights = None
    for epoch in range(1):
        print(args.src,args.pred)
        train_iter = data.iterator.pool(train_data,
                                        args.batchsize,
                                        #args.wbatchsize,
                                        key=lambda x: (len(x[0]), len(x[1])))

        '''dev_iter = data.iterator.pool(train_data,
                                      args.wbatchsize,
                                      key=lambda x: (len(x[0]), len(x[1])),
                                      batch_size_fn=batch_size_fn,
                                      random_shuffler=data.iterator.
                                      RandomShuffler())

        for dev_batch in dev_iter:
               print('Dev batch')
               model.eval()
               in_arrays = utils.seq2seq_pad_concat_convert(dev_batch, -1)
               print(len(dev_batch))
               _, stat = model(*in_arrays)
               #valid_stats.update(stat)'''

        #exit()
        #index = -2
        output_file = args.out_file
        output_fp = io.open(output_file, 'w', 1)
        org_vocab = vocab
        for num_steps, train_batch in enumerate(train_iter):
            # Run for 100 sentences
            if num_steps>49:
                 break
            print("Num steps:",num_steps)
            #print('length of train batch:',len(train_batch))
            in_arrays = utils.seq2seq_pad_concat_convert(train_batch, -1)
            #loss, stat = model(*in_arrays, index=index, vocab=vocab)
            loss, stat = model(*in_arrays)
            print(stat.loss_array.shape)
            if model.criterion.__class__.__name__=='KLDivLoss':
                print(stat.loss_array.sum(-1).mean(-1).item())
                print('original loss:', loss.item())
            if model.criterion.__class__.__name__=='NLLLoss':
                print(stat.loss_array.mean(-1).item())
                print('original loss:', loss.item())
            org_loss = loss.item()
            count = 0
            value = -1
            max_count = 10
            max_iteration = 1000
            threshold = 0.9
            min_loss = 100
            batch = train_batch
            size = int(args.size)
            # capping longer sentences keeping size to 75
            if len(train_batch[0][0])>40:
                print("Breaking the attack. Length of tran_batch is",len(train_batch[0][0]))
                #continue
                break
            ###a = model.embed_word.weight/model.embed_word.weight.norm(dim=1)[:,None]
            ###sim = torch.matmul(a,a.transpose(1,0))
            ###indices = np.unique(train_batch[0][0])
            ###print("indices:",len(indices))
            ###most_sim = torch.zeros(sim.shape[0],20,dtype=torch.int64).cuda()
            ###most_sim[indices] = torch.topk(sim[:,org_vocab][indices],20,dim=1)[1]
            ###sim_words = np.array(org_vocab)[np.array(torch.flatten(most_sim[indices]).cpu())]
            ###print("sim_words:",len(sim_words))
            vocab = org_vocab
            forbid_words = set([w2id[id2w[i].capitalize()] for i in train_batch[0][0] if id2w[i].capitalize() in good_words]).union(set([w2id[id2w[i].lower()] for i in train_batch[0][0] if id2w[i].lower() in w2id.keys()]).union(train_batch[0][0]))
            vocab = list(set(vocab).difference(set(train_batch[0][0])))
            vocab_new = list(set(vocab).difference(forbid_words))
            n_vocab = len(vocab)
            print(len(org_vocab), len(vocab), len(vocab_new), len(train_batch[0][0]))
            vocab = vocab_new
            print(len(vocab))
            n_vocab = len(vocab)
            ###vocab = list(set(vocab).difference(sim_words))
            ###print("Vocab:",len(vocab))
            ###n_vocab = len(vocab)
            ###del(sim)
            ###del(most_sim)
            #print(len(train_batch[0][0]))
            #zeros = np.zeros(len(train_batch[0][0])).astype(np.int)
            #train_batch[0] = (zeros, train_batch[0][1])
            index_changed = []
            for iter in range(5):
                flag = False
                for index in torch.randperm(len(train_batch[0][0])):
                    count = 0
                    value = -1
                    '''params = filter(lambda p: p.requires_grad, model.parameters())
                    optimizer = torch.optim.Adam(params,
                                                  lr=args.learning_rate,
                                                  betas=(args.optimizer_adam_beta1,
                                                         args.optimizer_adam_beta2),
                                                  eps=args.optimizer_adam_epsilon)'''
                        
                    losses = []
                    for start in range(0, n_vocab, size):
                        end = min(n_vocab, start+size)
                        length = min(size, n_vocab-start)
                        s = np.repeat(batch[0][0].reshape(1, -1), length, axis=0)
                        t = np.repeat(batch[0][1].reshape(1, -1), length, axis=0)
                        s[:, index] = vocab[start:end]
                        batch = list(zip(s, t))
                        #print(s.shape)
                        #print(t.shape)
                        in_arrays = utils.seq2seq_pad_concat_convert(batch, -1)
                        model.eval()
                        if len(args.multi_gpu) > 1:
                             loss_tuple, stat_tuple = zip(*dp(model, in_arrays, device_ids=args.multi_gpu))
                             n_total = sum([obj.n_words.item() for obj in stat_tuple])
                             n_correct = sum([obj.n_correct.item() for obj in stat_tuple])
                             loss = 0
                             for l_, s_ in zip(loss_tuple, stat_tuple):
                                  loss += l_ * s_.n_words.item()
                             loss /= n_total
                             stat = utils.Statistics(loss=loss.data.cpu() * n_total,
                                                     n_correct=n_correct,
                                                     n_words=n_total)
                             stat.loss_array = torch.cat([obj.loss_array for obj in stat_tuple],dim=0)
                        else:
                             loss, stat = model(*in_arrays)
                        #print("Shape of stat.loss_array:",stat.loss_array.shape)
                        #print("reshaped:",stat.loss_array.reshape(length,-1,len(id2w)).sum(-1).mean(-1).shape)
                        #print(stat.loss_array.reshape(length,-1,len(id2w)).sum(-1).mean(-1))
                        #print(loss)
                        #print(stat.loss_array.reshape(length,-1,len(id2w)).sum(-1).mean(-1).mean(-1))
                        stat.loss_array = stat.loss_array.detach()
                        if model.criterion.__class__.__name__=='KLDivLoss':
                            losses.append(stat.loss_array.reshape(length,-1,len(id2w)).sum(-1).mean(-1))
                        if model.criterion.__class__.__name__=='NLLLoss':
                            losses.append(stat.loss_array.reshape(length,-1).mean(-1))
                        #print(stat.loss_array.reshape(length, -1).mean(-1))
                        #print(len(losses), losses[-1].shape)
                        #print(torch.cuda.memory_allocated(device='cuda'))
                        #print("================================")
                        batch = train_batch
                    losses = torch.cat(losses, dim=0)
                    loss = losses.min()
                    new_word = losses.argmin()
                    #print(losses.shape)
                    #do_replace not needed for brute_force
                    old_word = batch[0][0][index]
                    # TODO: should be vocab[new_word] check
                    #print("Old and new word:",old_word,vocab[new_word])
                    if min_loss > loss and vocab[new_word]!=old_word:
                        flag = True
                        if index not in index_changed:
                             index_changed.append(index)
                        min_loss = max(loss.item(), org_loss)
                        #min_loss = loss.item()
                        src = train_batch[0][0]
                        # new_word = model.weights.argmax()
                        new_word = vocab[new_word]
                        src[index] = new_word
                        batch[0] = (src, batch[0][1])
                    print('loss({})\tmin_loss({})\tindex({})'.format(loss.item(), min_loss,index))
                if flag == False:
                    break
            src = batch[0][0]
            src_text = ' '.join([id2w[i]    for i in src]) + '\n'
            output_fp.write(src_text)
            print(index_changed)
            print(train_batch, src_text)
        output_fp.close()
            
if __name__ == '__main__':
    main()
