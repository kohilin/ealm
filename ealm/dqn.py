import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from collections import deque


from nn import MLP, key_value_attention, DEVICE


# Util functions
def gen_uniform_action_score(act_num, length):
    value = 1/act_num
    return torch.Tensor([value]).repeat(length, act_num)


def init_hiddens_and_scores(hiddens, scores):
    """ Generate initial representations """
    if scores is None:
        scores = gen_uniform_action_score(3, hiddens.shape[0])

    if isinstance(hiddens, np.ndarray):
        hiddens = torch.Tensor(hiddens)

    if isinstance(scores, np.ndarray):
        scores = torch.Tensor(scores)
    return hiddens.to(DEVICE), scores.to(DEVICE)


class LocalGlobalEncoder(nn.Module):
    def __init__(self, action_size=3, size=768):
        super().__init__()
        self.size = size * 2
        self.bias1 = nn.Embedding(action_size, 1)
        self.bias1.weight.data.uniform_(-1.0, 1.0)
        self.bias2 = nn.Embedding(2, 1)
        self.bias2.weight.data.uniform_(-1.0, 1.0)
        self.has_loss = False

    def forward(self, hiddens, scores=None, **kwargs):
        hiddens, scores = init_hiddens_and_scores(hiddens, scores)

        action_bias = self.bias1(kwargs.get("labels").to(DEVICE))
        acted_bias = self.bias2(kwargs.get("predicted").to(DEVICE))
        local_e = hiddens + action_bias + acted_bias
        global_e = key_value_attention(local_e, local_e)
        return torch.cat([local_e, global_e], dim=1)


class EditorialAgent(nn.Module):
    REMOVE, KEEP, REPLACE = 0, 1, 2

    def __init__(self, layer_num=2, hidden_dim=200):
        super().__init__()
        self.action_size = 3
        self.state_size = 768 * 2
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 0.900
        self.epsilon_min = 0.03
        self.epsilon_decay = 0.995
        self.temperature = 768
        self.selection_epsilon = 0.900
        self.selection_epsilon_min = 0.03
        self.selection_epsilon_decay = 0.995
        self.encoder = LocalGlobalEncoder()
        self.mlp = MLP(in_dim=self.state_size, hid_dim=hidden_dim,
                       out_dim=self.action_size, layer_num=layer_num)
        self.to(DEVICE)

    def predict(self, hidden, prev_score, do_exploration=True, **enc_kwargs):
        inp = self.encoder(hidden, prev_score, **enc_kwargs)
        if np.random.rand() <= self.epsilon and do_exploration:
            return torch.Tensor(np.random.rand(inp.shape[0], self.action_size)).to(DEVICE)
        else:
            scores = self.mlp(inp)
            return scores

    def replay(self, items):
        outputs, next_outputs, pred_idxs, actions, rewards, is_terminals = zip(*[(
            self.predict(i.hidden, i.prev_score, do_exploration=False,
                         **{"labels": torch.tensor(i.prev_labels),
                            "predicted": torch.tensor(i.prev_is_predicted),
                            "temperature": self.temperature}),
            i.next_max_score,
            i.pred_idx,
            i.action,
            i.reward,
            i.is_terminal
        ) for i in items])

        values = [o[i] for (o, i) in zip(outputs, pred_idxs)]

        values = torch.stack(values).to(DEVICE)
        rewards = torch.Tensor(rewards).to(DEVICE)
        actions = torch.tensor(np.stack(actions)).to(DEVICE)
        next_max_scores = torch.Tensor(np.stack(next_outputs)).to(DEVICE)

        action_values = values.gather(1, actions.unsqueeze(1)).squeeze(1)

        next_action_values = next_max_scores

        expected_action_values = rewards + self.gamma * next_action_values

        loss = F.smooth_l1_loss(action_values, expected_action_values)

        return loss, rewards.mean()

    def replay_encoder(self, items):
        encoder_loss = torch.stack([self.encoder.loss(i.hidden, i.prev_score) for i in items]).sum()
        return encoder_loss.sum()

    def apply_epsilon_decay(self):
        # update exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if self.selection_epsilon_min < self.selection_epsilon:
            self.selection_epsilon *= self.selection_epsilon_decay

    def prepare(self, hidden):
        length = len(hidden)
        is_predicted = torch.tensor([0] * length)
        labels = torch.tensor([1] * length)  # default keep all
        labels_onehot = torch.FloatTensor(length, 3)
        labels_onehot.zero_()
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
        return is_predicted, labels, labels_onehot

    def get_highest_entropy_idx(self, scores, is_predicted):
        p = scores / scores.sum(dim=1).unsqueeze(1)
        h = (p * p.log()).sum(dim=1) * -1
        h[is_predicted == 1] = -np.Inf
        sorted_idx = torch.argsort(h)
        max_h = sorted_idx[-1]
        return max_h

    def get_highest_value_idx(self, scores, is_predicted):
        maxs = scores.max(dim=1)[0]
        maxs[is_predicted == 1] = -np.Inf
        sorted_idx = torch.argsort(maxs)
        max_v = sorted_idx[-1]
        return max_v

    def get_ordered_idx_and_score(self, scores, is_predicted):
        scores_flat = scores.clone()
        scores_flat[is_predicted == 1] = -np.Inf
        scores_flat = scores_flat.view(-1)
        sorted_idxs = torch.argsort(scores_flat, descending=True)
        row_idxs = sorted_idxs / 3
        col_idxs = sorted_idxs % 3
        return [{"idx": r, "action": c, "score": s} for (r, c, s) in zip(row_idxs, col_idxs, scores_flat[sorted_idxs])]

    def get_action_at(self, scores, idx):
        action_at = torch.argmax(scores[idx])
        one_hot = torch.zeros(3)
        one_hot[action_at] = 1
        return action_at, one_hot

    def get_next_prev_score(self, scores, is_predicted, onehot_labels):
        scores_ = scores.clone()
        for idx, i in enumerate(is_predicted):
            if i == 0:
                scores_[idx] = scores_[idx]
            else:
                scores_[idx] = onehot_labels[idx]
        return scores_

    def npfy(self, x):
        return x.cpu().numpy() if isinstance(x, torch.Tensor) else x

    def process(self, sentence, hidden, do_exploration=True):
        with torch.no_grad():
            is_predicted, labels, labels_onehot = self.prepare(hidden)
            items, labels_per_step = [], []
            prev_score, next_score, prev_labels, prev_is_predicted = None, None, None, None

            for i in range(len(sentence)):
                # set previous score
                prev_score = next_score
                prev_labels = labels.clone()
                prev_is_predicted = is_predicted.clone()

                # compute current score
                cur_scores = self.predict(hidden, prev_score, do_exploration, **{"labels": prev_labels, "predicted": prev_is_predicted, "temperature": self.temperature})
                if i != 0:
                    items[-1].next_max_score = self.npfy(torch.max(cur_scores, dim=1)[0][is_predicted != 1].max())

                # get target index to predict next
                if do_exploration and np.random.rand() <= self.selection_epsilon:
                    target_idx = self.get_highest_entropy_idx(cur_scores, is_predicted)
                else:
                    target_idx = self.get_highest_value_idx(cur_scores, is_predicted)

                # update predicted indices
                is_predicted[target_idx] = 1

                # get predicted action at the index in scalar and one-hot formats
                action, action_onehot = self.get_action_at(cur_scores, target_idx)

                # update labels
                labels[target_idx] = action
                labels_onehot[target_idx] = action_onehot
                labels_per_step.append(labels.clone())

                # prepare prev score in next step
                next_score = self.get_next_prev_score(cur_scores, is_predicted, labels_onehot)

                # prepare an episode tuple
                item_args = map(self.npfy, [sentence, hidden, prev_score, cur_scores, next_score, target_idx, action, labels.clone(), prev_labels, prev_is_predicted, False])
                items.append(Item(*item_args))

            # assert all([i == 1 for i in is_predicted])
            items[-1].is_terminal = True
            return items


class Item:
    def __init__(self, sentence, hidden, prev_score, cur_score, next_score,
                 pred_idx, action, labels, prev_labels, prev_is_predicted, is_terminal):
        self.sentence = sentence
        self.hidden = hidden
        self.prev_score = prev_score
        self.cur_score = cur_score
        self.next_score = next_score
        self.next_max_score = None  # for fixed target-q, to be fixed
        self.pred_idx = pred_idx
        self.action = action
        self.labels = labels
        self.prev_labels = prev_labels
        self.prev_is_predicted = prev_is_predicted
        self.is_terminal = is_terminal
        self.reward = None
        self.cr = None
        self.rr = None
        self.crr = None

        self.comp_sent = None
        self.comp_sent_topk = None
        self.recon_sent = None
        self.recon_sent_topk = None
        self.comp_llh = None
        self.comp_sim = None

    def set(self, cs, csk, rs, rsk, llh, sim):
        self.comp_sent = cs
        self.comp_sent_topk = list(map(set, csk))
        self.recon_sent = rs
        self.recon_sent_topk = list(map(set, rsk))
        self.comp_llh = llh
        self.comp_sim = sim

    def report(self):
        cur_score = "[{:+06.2f}, {:+06.2f}, {:+06.2f}]".format(*list(self.cur_score[self.pred_idx]))
        return "cr={:.2f}/rr={:.2f}/crr={:.2f}/llh={:.2f}/sim={:.2f}/act={}({:10}, {}) -> {:+.2f} : {}".format(
            self.cr, self.rr, self.cr + self.rr,
            self.comp_llh, self.comp_sim, self.action, self.sentence[self.pred_idx], cur_score,
            self.reward, " ".join(self.comp_sent))

    def is_bad(self):
        if self.reward == -1 and np.random.rand() < 0.5:
            return True
        return False


class Memory:
    def __init__(self, buffer_size=2000):
        self.memory = deque()
        self.buffer_size = buffer_size
        self.mappend = self.memory.append
        self.mpopleft = self.memory.popleft

    def __call__(self, *args, **kwargs):
        return self.memory

    def size(self):
        return len(self.memory)

    def append(self, x):
        if len(self.memory) == self.buffer_size:
            self.mpopleft()
        self.mappend(x)

    def get(self):
        return self.memory

    def sample(self, n):
        idxs = np.random.randint(0, self.size(), n)
        return [self.memory[i] for i in idxs]

    def averaged_reward(self):
        return sum([i.reward for i in self.memory])/self.size()

