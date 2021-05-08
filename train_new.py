# encoding: utf-8
from __future__ import unicode_literals, print_function

import json
import os
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
import logging
from torch.autograd import Variable

import evaluator
from models import MultiTaskNMT, Transformer
from exp_moving_avg import ExponentialMovingAverage
from optimizer import NoamAdamTrainer, Yogi
from torchtext import data
import utils
from config import get_train_args
from fp16_utils import FP16_Optimizer, FP16_Module
from data_parallel import data_parallel as dp
from copy import deepcopy
from collections import Counter

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


def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger


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
                report_every,adv=False):
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
        #if adv:
        #     print("Adversarial")
        report_stats.output(epoch, batch + 1, num_batches, start_time)
        report_stats = utils.Statistics()
    return report_stats


# Have to unwrap DDP & FP16, if using.
def unwrap(module, model_name='Transformer'):
    if isinstance(module, eval(model_name)):
        return module
    return unwrap(module.module, model_name)


class CalculateBleu(object):
    def __init__(self, model, test_data, key, batch=50, max_decode_len=50,
                 beam_size=1, alpha=0.6, max_sent=None):
        self.model = unwrap(model)
        self.test_data = test_data
        self.key = key
        self.batch = batch
        self.device = -1
        self.max_decode_length = max_decode_len
        self.beam_size = beam_size
        self.alpha = alpha
        self.max_sent = max_sent

    def __call__(self, logger):
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
            try:
                ys = self.model.translate(x_block,
                                      self.max_decode_length,
                                      beam=self.beam_size,
                                      alpha=self.alpha)
            except:
                '''checkpoint = torch.load(args.best_model_file)
                save_checkpoint({'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'state_dict_ema': ema.shadow_variable_dict,
                        'best_score': best_score,
                        'optimizer': optimizer.state_dict(),
                        'opts': args,
                    },  is_best,
                        args.model_file,
                        args.best_model_file)'''
                print("model saved")
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
        logger.info('BLEU: {}'.format(bleu.score_str()))
        logger.info('')
        return bleu.bleu, hypotheses


def main():
    best_score = 0
    args = get_train_args()
    logger = get_logger(args.log_path)
    logger.info(json.dumps(args.__dict__, indent=4))

    # Set seed value
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.gpu:
        torch.cuda.manual_seed_all(args.seed)

    # Reading the int indexed text dataset
    train_data = np.load(os.path.join(args.input, args.data + ".train.npy"), allow_pickle=True)
    train_data = train_data.tolist()
    dev_data = np.load(os.path.join(args.input, args.data + ".valid.npy"), allow_pickle=True)
    dev_data = dev_data.tolist()
    test_data = np.load(os.path.join(args.input, args.data + ".test.npy"), allow_pickle=True)
    test_data = test_data.tolist()

    # Reading the vocab file
    with open(os.path.join(args.input, args.data + '.vocab.pickle'), 'rb') as f:
        id2w = pickle.load(f)

    w2id = {word: index for index, word in id2w.items()}
    args.id2w = id2w
    args.n_vocab = len(id2w)
    good_words, good_vocab = torch.load(os.path.join(args.input, 'good_words_in_vocab.pt'))
    # Define Model
    model = eval(args.model)(args)
    model.apply(init_weights)

    tally_parameters(model)
    if args.gpu >= 0:
        model.cuda(args.gpu)
    logger.info(model)

    if args.optimizer == 'Noam':
        optimizer = NoamAdamTrainer(model, args)
    elif args.optimizer == 'Adam':
        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(params,
                                     lr=args.learning_rate,
                                     betas=(args.optimizer_adam_beta1,
                                            args.optimizer_adam_beta2),
                                     eps=args.optimizer_adam_epsilon)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='max',
                                                               factor=0.7,
                                                               patience=7,
                                                               verbose=True)
    elif args.optimizer == 'Yogi':
        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = Yogi(params,
                         lr=args.learning_rate,
                         betas=(args.optimizer_adam_beta1,
                                args.optimizer_adam_beta2),
                         eps=args.optimizer_adam_epsilon)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='max',
                                                               factor=0.7,
                                                               patience=7,
                                                               verbose=True)

    if args.fp16:
        model = FP16_Module(model)
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=args.static_loss_scale,
                                   dynamic_loss_scale=args.dynamic_loss_scale,
                                   dynamic_loss_args={'init_scale': 2 ** 16},
                                   verbose=False)
        
    ema = ExponentialMovingAverage(decay=args.ema_decay)
    ema.register(model.state_dict())

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.model_file):
            logger.info("=> loading checkpoint '{}'".format(args.model_file))
            checkpoint = torch.load(args.model_file)
            args.start_epoch = checkpoint['epoch']
            best_score = checkpoint['best_score']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".
                  format(args.model_file, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.model_file))

    src_data, trg_data = list(zip(*train_data))
    total_src_words = len(list(itertools.chain.from_iterable(src_data)))
    total_trg_words = len(list(itertools.chain.from_iterable(trg_data)))
    iter_per_epoch = (total_src_words + total_trg_words) // (2 * args.wbatchsize)
    logger.info('Approximate number of iter/epoch = {}'.format(iter_per_epoch))
    time_s = time()

    global_steps = 0
    num_grad_steps = 0
    if args.grad_norm_for_yogi and args.optimizer == 'Yogi':
        args.start_epoch = -1
        l2_norm = 0.0
        parameters = list(filter(lambda p: p.requires_grad is True, model.parameters()))
        n_params = sum([p.nelement() for p in parameters])
    '''rep_d = Counter()
    print("preparing similarity matrix")
    org_data = torch.load("./temp/run_en_de_new/models/model_best_run_en_de_new.ckpt")
    org_data = org_data['state_dict']
    org_emb_W = org_data['embed_word.weight']
    a = org_emb_W/org_emb_W.norm(dim=1)[:,None]
    sim = torch.matmul(a.cpu(),a.cpu().transpose(1,0))
    org_vocab = pickle.load(open("./temp/run_en_de_new/data/processed.vocab.pickle","rb"))
    org_good_words = torch.load("./temp/run_en_de_new/data/good_words_in_vocab.pt")
    org_most_sim = torch.topk(sim[:,org_good_words[1]],20,dim=1)[1]
    org_most_sim = np.array(org_good_words[1])[org_most_sim]
    print("org_most_sim[11]",org_most_sim[11])
    most_sim = np.zeros((len(org_vocab),20),dtype='int64')
    for i in range(len(org_vocab)):
         idx = w2id[org_vocab[i]]
         most_sim[idx] = np.array([w2id[org_vocab[j]] for j in org_most_sim[i]])
    print("Doing sanity check")
    for i in range(len(org_vocab)):
         idx = w2id[org_vocab[i]]
         l = [org_vocab[k] for k in org_most_sim[i]]
         l1 = [id2w[k] for k in most_sim[idx]]
         assert np.array_equal(np.array(l),np.array(l1))
    print("sanity check passed")
    print("most_sim",most_sim)'''
    for epoch in range(args.start_epoch, args.epoch):
        random.shuffle(train_data)
        train_iter = data.iterator.pool(train_data,
                                        args.wbatchsize,
                                        key=lambda x: (len(x[0]), len(x[1])),
                                        batch_size_fn=batch_size_fn,
                                        sort_within_batch=True,
                                        random_shuffler=
                                        data.iterator.RandomShuffler())
        report_stats = utils.Statistics()
        train_stats = utils.Statistics()
        adv_report_stats = utils.Statistics()
        adv_train_stats = utils.Statistics()
        if epoch==0:
            np.save("train_data_ep_0.npy",train_data)
        if epoch==1:
            np.save("train_data_ep_1.npy",train_data)
        if args.debug:
            grad_norm = 0.
        for num_steps, train_batch in enumerate(train_iter):
            global_steps += 1
            model.train()
            if args.grad_accumulator_count == 1:
                optimizer.zero_grad()
            elif num_grad_steps % args.grad_accumulator_count == 0:
                optimizer.zero_grad()
            src_iter = list(zip(*train_batch))[0]
            src_words = len(list(itertools.chain.from_iterable(src_iter)))
            report_stats.n_src_words += src_words
            train_stats.n_src_words += src_words
            adv_report_stats.n_src_words += src_words
            adv_train_stats.n_src_words += src_words
            in_arrays = utils.seq2seq_pad_concat_convert(train_batch, -1)
            if len(args.multi_gpu) > 1:
                loss_tuple, stat_tuple, logit_tuple = zip(*dp(model, in_arrays, device_ids=args.multi_gpu))
                n_total = sum([obj.n_words.item() for obj in stat_tuple])
                n_correct = sum([obj.n_correct.item() for obj in stat_tuple])
                loss = 0
                for l_, s_ in zip(loss_tuple, stat_tuple):
                    loss += l_ * s_.n_words.item()
                loss /= n_total
                stat = utils.Statistics(loss=loss.data.cpu() * n_total,
                                         n_correct=n_correct,
                                         n_words=n_total)
                logits = torch.cat([obj for obj in logit_tuple],dim=0)
            else:
                loss, stat, logits = model(*in_arrays,return_logits=True)
            print("Clean loss:",loss)
            #print("logits:",logits)
            #print("logits.argmax",logits.argmax(dim=1))
            #print("batch size:",len(train_batch))
            #print("target:",in_arrays[2],in_arrays[2].shape)
            #print("logits argmax reshape:",logits.argmax(dim=1).reshape(in_arrays[2].shape))
            logits = logits.argmax(dim=1).reshape(in_arrays[2].shape)
            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
            if epoch == -1 and args.grad_norm_for_yogi and args.optimizer == 'Yogi':
                l2_norm += (utils.grad_norm(model.parameters()) ** 2) / n_params
                continue
            num_grad_steps += 1
            if args.debug:
                norm = utils.grad_norm(model.parameters())
                grad_norm += norm
                if global_steps % args.report_every == 0:
                    logger.info("> Gradient Norm: %1.4f" % (grad_norm / (num_steps + 1)))
            if args.grad_accumulator_count == 1:
                optimizer.step()
                ema.apply(model.state_dict(keep_vars=True))
            elif num_grad_steps % args.grad_accumulator_count == 0:
                optimizer.step()
                ema.apply(model.state_dict(keep_vars=True))
                num_grad_steps = 0
            report_stats.update(stat)
            train_stats.update(stat)
            report_stats = report_func(epoch, num_steps, iter_per_epoch,
                                       time_s, report_stats, args.report_every)

            valid_stats = utils.Statistics()
            if epoch>=int(args.epoch_start):
                if epoch==int(args.epoch_start) and num_steps==0:
                    rep_d = Counter()
                    init_word_d = Counter()
                    print("Calculating similarity matrix")
                    a = model.embed_word.weight/model.embed_word.weight.norm(dim=1)[:,None]
                    sim = torch.matmul(a,a.transpose(1,0))
                    print("Calculating topk")
                    most_sim = torch.topk(sim[:,good_vocab],int(args.emb_topk),dim=1)[1]
                    most_sim = np.array(good_vocab)[most_sim.cpu()]
                    print("Freezing embedding matrix")
                    print("require_grad for embed:", model.embed_word.weight.requires_grad)
                    model.embed_word.weight.requires_grad = False
                    print("require_grad for embed:", model.embed_word.weight.requires_grad)
                    print("saving word embedding")
                    np.save(os.path.dirname(args.input)+"/embedding.npy",np.array(model.embed_word.weight.cpu().numpy()))
                    del(sim)
                if num_steps==0:
                    print("length of rep_d:",len(rep_d))
                    print("emb_W:",model.embed_word.weight[10][:50])
                    #np.save(os.path.dirname(args.input)+"/embedding-epoch-%d.npy"%epoch,np.array(model.embed_word.weight.cpu().numpy()))
                # New code
                optimizer.steps -= 1
                if args.grad_accumulator_count == 1:
                    optimizer.zero_grad()
                elif num_grad_steps % args.grad_accumulator_count == 0:
                    optimizer.zero_grad()
                #frac_replace = 0.3 # fraction of words to replace
                frac_replace = float(args.frac_replace)
                train_batch_new = deepcopy(train_batch)
                #print("calculating similarity")
                ###a = model.embed_word.weight/model.embed_word.weight.norm(dim=1)[:,None]
                ###sim = torch.matmul(a,a.transpose(1,0))
                #print("calculating topk")
                #most_sim = torch.topk(sim[:,good_vocab],20,dim=1)[1]
                ###indices = np.concatenate([train_batch_new[i][0] for i in range(len(train_batch_new))])
                ###indices = np.unique(indices)
                ###print("indices:",len(indices))
                ###most_sim = torch.zeros(sim.shape[0],20,dtype=torch.int64).cuda()
                ###most_sim[indices] = torch.topk(sim[:,good_vocab][indices],20,dim=1)[1]
                ###print("done")
                #print("target before:",in_arrays[2])
                #print("logits:",logits)
                logits = logits.cpu().numpy()
                t0 = time()
                for i in range(len(train_batch_new)):
                    if len(np.where(logits[i]==1)[0])==1 and logits[i][0]!=1:
                        ## Truncate till <eos> token (<eos> is idx 1 in vocab)
                        #print("truncating till <eos>")
                        train_batch_new[i][1] = logits[i][:np.where(logits[i]==1)[0][0]]
                        #print("changed to",train_batch_new[i][1])
                    else:
                        train_batch_new[i][1] = logits[i]
                    num_words = np.ceil(train_batch_new[i][0].shape[0]*frac_replace).astype('int')
                    if i==0:
                        print("num_words:",num_words)
                    ## Unbiased position selection becomes biased toward high frequency words(commenting for now)
                    #inds = np.random.permutation(range(train_batch_new[i][0].shape[0]))[:num_words]
                    ## TODO: Counter the high frequency words bias using init_word_d
                    #print("Selecting positions")
                    cand_counts = [init_word_d[train_batch_new[i][0][j]] for j in range(len(train_batch_new[i][0]))]
                    #print("Counts and number of candidates:",cand_counts,len(cand_counts))
                    cand_logits = list(map(lambda i: np.exp(-1e-2*i) + 1e-9,cand_counts))## 1e-9 for numerical stability
                    cand_probs = cand_logits/np.sum(cand_logits)
                    #print("Probabilies:",cand_probs)
                    inds = np.random.choice(len(train_batch_new[i][0]),num_words, replace=False, p = cand_probs)
                    #print("Count of position indices chosen:",np.array(cand_counts)[inds])
                    #train_batch_new[i][0][inds] = np.random.choice(range(5,len(id2w)),num_words)
                    #vocab_prune = list(set(good_vocab).difference(train_batch[i][0]))
                    forbid_words = set([w2id[id2w[j].capitalize()] for j in train_batch[i][0] if id2w[j].capitalize() in good_words]).union(set([w2id[id2w[j].lower()] for j in train_batch[i][0] if id2w[j].lower() in w2id.keys()]).union(train_batch[i][0]))
                    vocab_prune = list(set(good_vocab).difference(forbid_words))

                    #train_batch_new[i][0][inds] = np.random.choice(good_vocab,num_words)
                    #train_batch_new[i][0][inds] = np.random.choice(vocab_prune,num_words)
                    x = most_sim[train_batch_new[i][0][inds]]
                    #print("train batch inds:",train_batch_new[i][0][inds])
                    #if len(inds)>1:
                    #    print("x,inds,old_words:",x,inds,train_batch_new[i][0][inds])
                    #    print("Old sentence:",train_batch_new[i][0])
                    ###out = x.transpose(1,0)[torch.randperm(20)][0]
                    ###out = x.transpose(1,0)[list(zip(*[(np.random.randint(20),i) for i in range(x.shape[0])]))]
                    ###train_batch_new[i][0][inds] = np.array(good_vocab)[list(map(lambda o:o.data.item(),out))]
                    ###train_batch_new[i][0][inds] = np.array(good_vocab)[np.array(out)]
                    forbid_words = np.array(list(forbid_words))
                    ###x_act = np.array(good_vocab)[np.array(x.cpu())]
                    x_act = x
                    x_act[np.where(np.isin(x_act,forbid_words))] = -1
                    ## Unbiased way (commenting for now)
                    #train_batch_new[i][0][inds] = np.array([np.random.choice(x_act[k][x_act[k]!=-1]) for k in range(len(x_act))])
                    # Trying an alternate way (biased towards less count)
                    for k, ind in enumerate(inds):
                        #print("Index:",ind)
                        #print("Initial word:",train_batch[i][0][ind])
                        cand_counts = [rep_d[(train_batch[i][0][ind], cand)] for cand in x_act[k][x_act[k]!=-1]]
                        #print("Counts and number of candidates:",cand_counts,len(cand_counts))
                        cand_logits = list(map(lambda i: np.exp(-1e-2*i) + 1e-9,cand_counts))## 1e-9 for numerical stability
                        cand_probs = cand_logits/np.sum(cand_logits)
                        #print("Probabilies:",cand_probs)
                        train_batch_new[i][0][ind] = np.random.choice(x_act[k][x_act[k]!=-1], p = cand_probs)
                        #print("x_act allowed:",x_act[k][x_act[k]!=-1])
                        #print("Final word:",train_batch_new[i][0][ind])
                        #print(np.where(x_act[k][x_act[k]!=-1]==train_batch_new[i][0][ind]))
                        #print("Count of word selected",np.array(cand_counts)[np.where(x_act[k][x_act[k]!=-1]==train_batch_new[i][0][ind])])
                    ###print(np.array(list(zip(train_batch[i][0][inds],train_batch_new[i][0][inds]))))
                    ###print("np_rep is:")
                    ###if len(np_rep)==1:
                    ###     np_rep = np.array(list(zip(train_batch[i][0][inds],train_batch_new[i][0][inds])))
                    ###else:
                    ###     np_rep = np.concatenate((np_rep,np.array(list(zip(train_batch[i][0][inds],train_batch_new[i][0][inds])))))
                    ###np_rep = np.unique(np_rep,axis=0)
                    ###print(len(np_rep))
                    # This is slower
                    #rep_d += Counter(list(zip(train_batch[i][0][inds],train_batch_new[i][0][inds])))
                    # This is faster
                    for (k1,k2) in list(zip(train_batch[i][0][inds],train_batch_new[i][0][inds])):
                        rep_d[(k1,k2)]+=1
                        init_word_d[k1]+=1
                    ###print(len(rep_d))
                    ###for k in range(len(inds)):
                    ###    while train_batch_new[i][0][inds[k]] not in vocab_prune:
                    ###            print("Inside while")
                    ###         x = most_sim[train_batch[i][0][inds[k]]]
                    ###         #print("most similar:",x)
                    ###         out = x[torch.randperm(20)][0]
                    ###         ###out = x.transpose(1,0)[torch.randperm(20)][0]
                    ###         ###out = x.transpose(1,0)[torch.randperm(20)][:,k][0]
                    ###         ###print("most similar:",x.transpose(1,0)[torch.randperm(20)][:,k])
                    ###         train_batch_new[i][0][inds[k]] = np.array(good_vocab)[out.data.item()]
                    ###         #print(train_batch_new[i][0],forbid_words)
                    ###         #print("out selected:",out.data.item())
                    #if len(inds)>1:
                    #    print("new_words:",train_batch_new[i][0][inds])
                    #    print("New sentence:",train_batch_new[i][0])
                ###del(sim)
                ###del(most_sim)
                t1 = time()
                print("Time:",(t1-t0))
                in_arrays = utils.seq2seq_pad_concat_convert(train_batch_new, -1)
                #print("target after:",in_arrays[2])
                if len(args.multi_gpu) > 1:
                    loss_tuple, stat_tuple, logits_adv_tuple = zip(*dp(model, in_arrays, device_ids=args.multi_gpu))
                    n_total = sum([obj.n_words.item() for obj in stat_tuple])
                    n_correct = sum([obj.n_correct.item() for obj in stat_tuple])
                    loss = 0
                    for l_, s_ in zip(loss_tuple, stat_tuple):
                        loss += l_ * s_.n_words.item()
                    loss /= n_total
                    adv_stat = utils.Statistics(loss=loss.data.cpu() * n_total,
                                         n_correct=n_correct,
                                         n_words=n_total)
                    logits_adv = torch.cat([obj for obj in logits_adv_tuple],dim=0)

                else:
                    loss, adv_stat,logits_adv = model(*in_arrays,return_logits=True)
                '''print("Loss before:",loss)
                print("Stat before:",adv_stat.loss,adv_stat.n_words,adv_stat.loss/adv_stat.n_words)'''
                #loss = torch.clamp(loss,0.0,10.0)
                print("Noisy loss:",num_steps,loss)
                logits_adv = logits_adv.argmax(dim=1).reshape(in_arrays[2].shape)
                #print("logits:",logits[0])
                #print("logits_adv:",logits_adv[0])
                if loss.data.item()>float(args.clip):
                    for i in range(len(train_batch)):
                        for j in range(len(train_batch[i][0])):
                            if rep_d[(train_batch[i][0][j],train_batch_new[i][0][j])]>0:
                                #print("reducing count")
                                rep_d[(train_batch[i][0][j],train_batch_new[i][0][j])]-=1
                                init_word_d[train_batch[i][0][j]]-=1
                loss = torch.clamp(loss,0.0,float(args.clip))
                #print("Noisy loss:",num_steps,loss)
                adv_stat.loss = loss*adv_stat.n_words
                '''print("Loss:",loss)
                print("Stat:",adv_stat.loss,adv_stat.n_words,adv_stat.loss/adv_stat.n_words)'''
                if args.fp16:
                    #optimizer.backward(-0.1*loss)
                    optimizer.backward(-1*float(args.lambda_val)*loss)
                else:
                    #(-0.1*loss).backward()
                    (-1*float(args.lambda_val)*loss).backward()
                if epoch == -1 and args.grad_norm_for_yogi and args.optimizer == 'Yogi':
                    l2_norm += (utils.grad_norm(model.parameters()) ** 2) / n_params
                    continue
                num_grad_steps += 1
                if args.debug:
                    norm = utils.grad_norm(model.parameters())
                    grad_norm += norm
                    if global_steps % args.report_every == 0:
                        logger.info("> Gradient Norm: %1.4f" % (grad_norm / (num_steps + 1)))
                if args.grad_accumulator_count == 1:
                    optimizer.step()
                    ema.apply(model.state_dict(keep_vars=True))
                elif num_grad_steps % args.grad_accumulator_count == 0:
                    optimizer.step()
                    ema.apply(model.state_dict(keep_vars=True))
                    num_grad_steps = 0
                #print("before:", emb_W_before[0])
                #print("after:",emb_W_after[0])
                emb_W = model.embed_word.weight
                #print("emb_W",emb_W[10][:50])
                #print("grad:",list(model.parameters())[-1].grad)
                adv_report_stats.update(adv_stat)
                adv_train_stats.update(adv_stat)
                adv_report_stats = report_func(epoch, num_steps, iter_per_epoch,
                                                   time_s, adv_report_stats, args.report_every,adv=True)
            if global_steps % args.eval_steps == 0:
                '''dev_iter = data.iterator.pool(dev_data,
                                              args.wbatchsize,
                                              key=lambda x: (len(x[0]), len(x[1])),
                                              batch_size_fn=batch_size_fn,
                                              random_shuffler=data.iterator.
                                              RandomShuffler())

                for dev_batch in dev_iter:
                    model.eval()
                    in_arrays = utils.seq2seq_pad_concat_convert(dev_batch, -1)
                    if len(args.multi_gpu) > 1:
                        _, stat_tuple = zip(*dp(model, in_arrays, device_ids=args.multi_gpu))
                        n_total = sum([obj.n_words.item() for obj in stat_tuple])
                        n_correct = sum([obj.n_correct.item() for obj in stat_tuple])
                        dev_loss = sum([obj.loss for obj in stat_tuple])
                        stat = utils.Statistics(loss=dev_loss,
                                                n_correct=n_correct,
                                                n_words=n_total)
                    else:
                        _, stat = model(*in_arrays)
                    valid_stats.update(stat)'''

                logger.info('Train perplexity: %g' % train_stats.ppl())
                logger.info('Train accuracy: %g' % train_stats.accuracy())
                
                if epoch>=int(args.epoch_start):
                    logger.info('Adversarial train perplexity: %g' % adv_train_stats.ppl())
                    logger.info('Adversarial train accuracy: %g' % adv_train_stats.accuracy())

                '''logger.info('Validation perplexity: %g' % valid_stats.ppl())
                logger.info('Validation accuracy: %g' % valid_stats.accuracy())'''

                if args.metric == "accuracy":
                    score = valid_stats.accuracy()
                elif args.metric == "bleu":
                    score, _ = CalculateBleu(model,
                                             dev_data,
                                             'Dev Bleu',
                                             batch=args.batchsize // 4,
                                             beam_size=args.beam_size,
                                             alpha=args.alpha,
                                             max_sent=args.max_sent_eval)(logger)

                    '''score, _ = CalculateBleu(model,
                                             train_data,
                                             'Train Bleu',
                                             batch=args.batchsize // 4,
                                             beam_size=args.beam_size,
                                             alpha=args.alpha,
                                             max_sent=args.max_sent_eval)(logger)'''

                # Threshold Global Steps to save the model
                if global_steps > 8000  and epoch>=int(args.epoch_start):
                    is_best = score > best_score
                    best_score = max(score, best_score)
                    if is_best:
                         pickle.dump(rep_d,open(os.path.dirname(args.input)+'/count.pkl','wb'))
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'state_dict_ema': ema.shadow_variable_dict,
                        'best_score': best_score,
                        'optimizer': optimizer.state_dict(),
                        'opts': args,
                    },  is_best,
                        args.model_file,
                        args.best_model_file)

                if args.optimizer == 'Adam' or args.optimizer == 'Yogi':
                    scheduler.step(score)

        if epoch == -1 and args.grad_norm_for_yogi and args.optimizer == 'Yogi':
            optimizer.v_init = l2_norm / (num_steps + 1)
            logger.info("Initializing Yogi Optimizer (v_init = {})".format(optimizer.v_init))

    # BLEU score on Dev and Test Data
    checkpoint = torch.load(args.best_model_file)
    logger.info("=> loaded checkpoint '{}' (epoch {}, best score {})".
          format(args.best_model_file,
                 checkpoint['epoch'],
                 checkpoint['best_score']))
    model.load_state_dict(checkpoint['state_dict'])

    logger.info('Dev Set BLEU Score')
    _, dev_hyp = CalculateBleu(model,
                               dev_data,
                               'Dev Bleu',
                               batch=args.batchsize // 4,
                               beam_size=args.beam_size,
                               alpha=args.alpha,
                               max_decode_len=args.max_decode_len)(logger)
    save_output(dev_hyp, id2w, args.dev_hyp)

    logger.info('Test Set BLEU Score')
    _, test_hyp = CalculateBleu(model,
                                test_data,
                                'Test Bleu',
                                batch=args.batchsize // 4,
                                beam_size=args.beam_size,
                                alpha=args.alpha,
                                max_decode_len=args.max_decode_len)(logger)
    save_output(test_hyp, id2w, args.test_hyp)

    # Loading EMA state dict
    model.load_state_dict(checkpoint['state_dict_ema'])
    logger.info('Dev Set BLEU Score')
    _, dev_hyp = CalculateBleu(model,
                               dev_data,
                               'Dev Bleu',
                               batch=args.batchsize // 4,
                               beam_size=args.beam_size,
                               alpha=args.alpha,
                               max_decode_len=args.max_decode_len)(logger)
    save_output(dev_hyp, id2w, args.dev_hyp + '.ema')

    logger.info('Test Set BLEU Score')
    _, test_hyp = CalculateBleu(model,
                                test_data,
                                'Test Bleu',
                                batch=args.batchsize // 4,
                                beam_size=args.beam_size,
                                alpha=args.alpha,
                                max_decode_len=args.max_decode_len)(logger)
    save_output(test_hyp, id2w, args.test_hyp + '.ema')


if __name__ == '__main__':
    main()
