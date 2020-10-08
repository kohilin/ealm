import json
import os
import sys
from collections import namedtuple
import torch
from itertools import chain
import time
import tqdm
import trainer
import reward
import util

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_json(dir):
    with open(os.path.join(dir, "config.json"), "r") as f:
        j = json.load(f)
    return namedtuple('conf', j.keys())(*j.values())


def load_model(dir):
    file = "best-reward-model.pt"
    return torch.load(os.path.join(dir, file), map_location=DEVICE)


def init(opt):
    import bertnlp
    import dqn

    conf = load_json(opt.m)
    model = load_model(opt.m)

    bertnlp.init()
    agent = dqn.EditorialAgent(layer_num=int(conf.nlayer), hidden_dim=int(conf.hdim))
    agent.load_state_dict(model)
    agent.eval()
    return bertnlp, agent, conf


def run(bertnlp, agent, texts):
    tokenized_texts, hiddens = bertnlp.apply_tokenize_and_get_hiddens(texts, rm_pad=True)
    items = agent.process(tokenized_texts[0], hiddens[0], do_exploration=False)
    tokenized_texts, labels = zip(*[(i.sentence, i.labels) for i in items])
    comp_sents, comp_topk, recon_sents, recon_topk = \
        bertnlp.apply_compression_and_reconstruction(tokenized_texts, labels)
    for i, cs, csk, rs, rsl in zip(items, comp_sents, comp_topk, recon_sents, recon_topk):
        i.set(cs, csk, rs, rsl, 0, 0)
    items = reward.calculate_comp_recon_rewards(items)

    max_crr, max_crr_item = 0, items[0]
    for i in items:
        if i.crr >= max_crr:
            max_crr = i.crr
            max_crr_item = i

    return [max_crr_item.comp_sent]


if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser("Prediction script", add_help=True)

    args.add_argument("-m", help="Model directory", required=True)
    args.add_argument("-f", help="Target file", default=None)
    args.add_argument("-o", help="Output file", default=None)

    opt = args.parse_args()
    bertnlp, agent, conf = init(opt)
    sys.stderr.write("Training configuration: {}\n".format(conf))
    s = time.time()
    texts = trainer.load_file(opt.f)
    results = []
    for t in tqdm.tqdm(texts):
        results.append(run(bertnlp, agent, [t]))

    f = open(opt.o, "w")

    results = list(chain.from_iterable(results))
    for res in results:
        f.write(" ".join(res) + "\n")
    f.close()

    sys.stderr.write("Elapsed Time={:.3f}ms".format(time.time()-s))
