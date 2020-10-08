import copy
import json
import os
import shutil
import sys
import torch

import numpy as np

from itertools import chain

import bertnlp
import dqn
import reward
import util


logger = None


def train(opt):
    SAVE_DIR = opt.save
    LATEST_MODEL = os.path.join(SAVE_DIR, "latest-model.pt")
    BEST_REWARD_MODEL = os.path.join(SAVE_DIR, "best-reward-model.pt")
    CONFIG_FILE = os.path.join(SAVE_DIR, "config.json")
    LOG_FILE = os.path.join(SAVE_DIR, "log.txt")

    global logger
    logger = util.init_logger(LOG_FILE)

    with open(CONFIG_FILE, "w") as f:
        json.dump(vars(opt), f)

    bertnlp.init()

    agent = dqn.EditorialAgent(layer_num=opt.nlayer, hidden_dim=opt.hdim)

    train_sents = load_file(opt.t)

    memory = dqn.Memory(buffer_size=int(opt.memory))
    optimizer = torch.optim.Adam(agent.parameters(), lr=opt.lr)
    sentence_batcher = make_batcher(train_sents, 1)
    n_sentence_iter = 0
    model_update_interval = 1000
    max_avg_reward = -1
    try:
        for epoch in range(int(opt.e)):
            try:
                cur_sentences = next(sentence_batcher)
            except StopIteration:
                n_sentence_iter += 1
                logger.info("Corpus Iteration {}".format(n_sentence_iter))
                sentence_batcher = make_batcher(train_sents, 1)
                cur_sentences = next(sentence_batcher)

            items, reports = generate_episode_items(bertnlp, agent,
                                                    cur_sentences,
                                                    min_cr=opt.min_cr,
                                                    min_rr=opt.min_rr)
            [memory.append(i) for i in chain.from_iterable(items) if not i.is_terminal]

            if memory.size() < opt.batch:
                continue

            loss, reward = agent.replay(memory.sample(opt.batch))
            loss_, reward_ = step(loss, reward, optimizer, agent)

            msg = "Report : Epoch={} Reward={:.3f} Loss={:.3f} Eps1={:.3f} Eps2={:.3f}\n".format(
                epoch, reward_, loss_, agent.epsilon, agent.selection_epsilon)
            msg += "=" * 70 + "\n\t" + "\n\t".join(
                [i.report() for i in reports[0]]) + "\n" + "=" * 70
            logger.info(msg)

            if epoch != 0 and epoch % model_update_interval == 0:
                logger.info("Update latest model@Iteration {}".format(n_sentence_iter))
                save(agent, LATEST_MODEL)
                averaged_reward = memory.averaged_reward()
                if averaged_reward > max_avg_reward:
                    max_avg_reward = averaged_reward
                    save(agent, BEST_REWARD_MODEL)
                    logger.info("Update best reward model@Iteration{}(Averaged Reward={:.5f})".format(n_sentence_iter, max_avg_reward))

            if epoch != 0 and epoch % opt.decay_interval == 0:
                agent.apply_epsilon_decay()

    except KeyboardInterrupt:
        logger.info("Terminating process ... ")
        logger.info("done!")


def generate_episode_items(bertnlp, agent, sentences, min_cr=0.5, min_rr=0.5, add_prefix=True, k=10):

    def run_compression_reconstruction(items):
        orig_sents, labels = zip(*[(i.sentence, i.labels) for i in items])
        orig_sents = copy.deepcopy(orig_sents)
        labels = copy.deepcopy(labels)

        comp_sents, comp_topk, recon_sents, recon_topk = \
            bertnlp.apply_compression_and_reconstruction(orig_sents, labels, add_prefix, k)

        llhs = []
        for cs in comp_sents:
            llhs.append(bertnlp.mrf_log_prob(cs, tokenized=True))

        sims = bertnlp.predict_similarity(orig_sents, comp_sents, tokenized=True)

        for i, cs, csk, rs, rsl, llh, sim in zip(items, comp_sents, comp_topk,
                                                 recon_sents, recon_topk, llhs,
                                                 sims):
            i.set(cs, csk, rs, rsl, llh, sim)
        return items

    tokenized_sents, hiddens = bertnlp.apply_tokenize_and_get_hiddens(sentences, rm_pad=True)

    items = [agent.process(s, h, do_exploration=True) for (s, h) in zip(tokenized_sents, hiddens)]
    items = [run_compression_reconstruction(i) for i in items]

    train_items, report_items = zip(*[reward.reward(ei, min_cr=min_cr, min_rr=min_rr) for ei in items])
    return train_items, report_items


def load_file(file):
    with open(file, "r") as f:
        return [l.strip().split() for l in f]


def make_batcher(sentences, batch_size, do_shuf=True):
    N = len(sentences)
    idx = list(range(N))
    if do_shuf:
        idx = np.random.permutation(idx)
    for i in range(0, N, batch_size):
        cur_idx = idx[i:i + batch_size]
        yield [sentences[j] for j in cur_idx]


def step(loss, reward, opt, agent):
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(agent.parameters(), 1)
    opt.step()
    return loss.detach().cpu().float().numpy(), reward.detach().cpu().float().numpy()


def save(model, save_path):
    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    import argparse

    args = argparse.ArgumentParser("Training script", add_help=True)

    args.add_argument("-t", help="Training Corpus", required=True, type=str)
    args.add_argument("-e", help="Epoch", default=10000000000000, type=int)
    args.add_argument("-min_cr", help="Minimum threshold for compression",
                      default=0.3, type=float)
    args.add_argument("-min_rr", help="Minimum threshold for reconstruction",
                      default=0.5, type=float)
    args.add_argument("-lr", help="Learning rate", default=0.001, type=float)
    args.add_argument("-nlayer", help="Number of layer in the agent MLP",
                      default=2, type=int)
    args.add_argument("-hdim",
                      help="Dimension of hidden layers in the agent MLP",
                      default=200, type=int)
    args.add_argument("-batch", help="Batch size", default=1, type=int)
    args.add_argument("-memory", help="Memory buffer size", default=2000,
                      type=int)
    args.add_argument("-decay_interval",
                      help="Epoch interval for e-greedy decay", default=10,
                      type=int)
    args.add_argument("-report_num", help="Number of report items", default=5,
                      type=int)
    args.add_argument("-save", help="Model directory", default="./model",
                      type=str)
    args.add_argument("-stopwords", help="List of stopwords", default=None,
                      type=str)

    opt = args.parse_args()
    # if os.path.exists(opt.save):
    #     print("{} already exists!!".format(opt.save))
    #     print("Are you sure to overwrite?[Yn]: ", end="")
    #     x = input()
    #     if x == "Y":
    #         shutil.rmtree(opt.save)
    #     else:
    #         sys.exit(1)
    # os.mkdir(opt.save)

    train(opt)
