import os
import numpy as np
from dqn import EditorialAgent


recon_stopwords_file = \
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "stopwords.txt")
with open(recon_stopwords_file) as f:
    for l in f:
        recon_stopwords = set([l.strip() for l in f])


def get_compression_ratio(item):
    co_ratio = len(
        [i for i in item.labels if i == EditorialAgent.REMOVE]) / len(item.labels)
    return co_ratio


def get_reconstruction_ratio(item):
    cor, denom = 0, 0
    for i in range(len(item.labels)):
        if item.sentence[i] in recon_stopwords:
            continue
        if item.sentence[i] == "[UNK]":
            continue
        cor += item.sentence[i] in item.recon_sent_topk[i]
        denom += 1
    return 0 if denom == 0 else cor / denom


def calculate_comp_recon_rewards(episode_items):
    for ei in episode_items:
        ei.cr = get_compression_ratio(ei)
        ei.rr = get_reconstruction_ratio(ei)
        ei.crr = ei.cr + ei.rr
    return episode_items


def summary_assessment(episode_items):
    episode_items = calculate_comp_recon_rewards(episode_items)
    last_item = episode_items[-1]
    llh_boundary = 0.005
    llh_reward = (1 if last_item.comp_llh > llh_boundary else 0)
    sim_reward = last_item.comp_sim

    r = last_item.rr * last_item.cr
    r += sim_reward * 0.1
    r += llh_reward * 0.1

    for ei in episode_items:
        ei.reward = r
    return episode_items, episode_items


def reward(episode_items, min_cr, min_rr):
    nstep = len(episode_items)
    min_rr_per_step = 1 - np.cumsum([(1 - min_rr) / nstep for i in range(0, nstep)])
    min_cr_per_step = np.cumsum([(min_cr / nstep) for i in range(0, nstep)])
    llh_boundary = 0.005

    episode_items = calculate_comp_recon_rewards(episode_items)

    learn_items, all_items = [], []
    prev_n = len(episode_items[0].sentence)
    failed = False
    for cur_min_rr, cur_min_cr, ei in zip(min_rr_per_step, min_cr_per_step, episode_items):
        step_cr = len(ei.comp_sent) / prev_n
        prev_n = len(ei.comp_sent)

        cr = (1 - step_cr) if ei.cr >= cur_min_cr else 0
        rr = 1 if ei.rr >= cur_min_rr else -1

        r_sr = -9.99
        if not failed:
            r_sr = rr * cr

        ei.reward = r_sr
        all_items.append(ei)

        if not failed:
            if ei.rr < cur_min_rr or ei.cr < cur_min_cr:
                ei.reward = -1
                learn_items.append(ei)
                failed = True
            else:
                learn_items.append(ei)

    reached_step = len(learn_items) / nstep
    last_item = learn_items[-1]

    comp_recon_rate = last_item.rr * last_item.cr
    sim_reward = last_item.comp_sim
    llh_reward = (1 if last_item.comp_llh > llh_boundary else 0)

    r_sa = \
        reached_step * (comp_recon_rate + sim_reward * 0.1 + llh_reward * 0.1)
    for li in learn_items:
        li.reward += r_sa

    return learn_items, all_items
