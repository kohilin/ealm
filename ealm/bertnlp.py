import semantic_text_similarity.models as smodel
from dqn import EditorialAgent
import torch

from pytorch_transformers import BertTokenizer, BertConfig, BertForMaskedLM

PAD_ID = 0
MASK_ID = 103
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
C = "[CLS]"
S = "[SEP]"
M = "[MASK]"
MAX_LENGTH = 512


config, tokenizer, model, sim_model = None, None, None, None


def init(maxlen=512):
    global config, tokenizer, model, sim_model, MAX_LENGTH
    MAX_LENGTH = maxlen

    bert_model_name = 'bert-base-uncased'
    config = BertConfig.from_pretrained(bert_model_name)
    config.output_hidden_states = True
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    model = BertForMaskedLM.from_pretrained(bert_model_name, config=config)
    model.to(DEVICE)
    model.eval()

    sim_model = smodel.WebBertSimilarity(device=DEVICE)


def put_cls_sep(sentences, tokenized=False):
    if tokenized:
        return [[C] + s + [S] for s in sentences]
    else:
        return ["{} {} {}".format(C, s, S) for s in sentences]


def pad(token_id_sequences, pad_id=PAD_ID):
    maxlen = max([len(s) for s in token_id_sequences])
    if maxlen > MAX_LENGTH:
        maxlen = MAX_LENGTH
    rtn = [tis + [pad_id] * (maxlen - len(tis)) for tis in token_id_sequences]
    return torch.tensor(rtn).to(DEVICE)


def encode_pad(sentences, add_cls_sep=True, tokenized=False):
    if add_cls_sep:
        sentences = put_cls_sep(sentences, tokenized)

    if tokenized:
        token_ids = list(map(tokenizer.convert_tokens_to_ids, sentences))
    else:
        token_ids = list(map(tokenizer.encode, sentences))
    padded_token_ids = pad(token_ids)
    return padded_token_ids.to(DEVICE)


def tokenize_and_put_ids_one(sentence):
    tokenized_sent = tokenizer.tokenize(sentence)
    tokenized_sent_id = list(map(tokenizer.convert_tokens_to_ids, tokenized_sent))
    return tokenized_sent, tokenized_sent_id


def tokenize_and_put_ids(sentences):
    return list(zip(*map(tokenize_and_put_ids_one, sentences)))


def run_bert(inputs, encoded=False, add_cls_sep=False, tokenized=True):
    with torch.no_grad():
        if not encoded:
            inputs = encode_pad(inputs, add_cls_sep=add_cls_sep, tokenized=tokenized)
        logits, hiddens = model(inputs)
        hiddens = hiddens[-1]  # last layer
        return logits, hiddens


def apply_tokenize_and_get_hiddens(sentences, rm_pad=True):
    tokenized_sentences = list(map(tokenizer.convert_ids_to_tokens, map(tokenizer.convert_tokens_to_ids, sentences)))
    _, hiddens = run_bert(tokenized_sentences, add_cls_sep=True)

    hiddens = hiddens[:, 1:-1, :]
    if rm_pad:
        hiddens = [h[:len(s), :] for (s, h) in zip(tokenized_sentences, hiddens)]
    return tokenized_sentences, hiddens


def convert_to_masked_input(tokens, labels, comp_or_recon="comp", prefix_tokens=None):
    masked, non_masked = [], []
    if comp_or_recon == "recon":
        for (t, l) in zip(tokens, labels):
            non_masked.append(t)
            if l == EditorialAgent.REMOVE:
                masked.append("[MASK]")
            elif l == EditorialAgent.KEEP:
                masked.append(t)
            elif l == EditorialAgent.REPLACE:
                # predict tokens
                masked.append("[MASK]")
            else:
                assert False

    elif comp_or_recon == "comp":
        for (t, l) in zip(tokens, labels):
            if l == EditorialAgent.REMOVE:
                continue
            elif l == EditorialAgent.KEEP:
                non_masked.append(t)
                masked.append(t)
            elif l == EditorialAgent.REPLACE:
                non_masked.append(t)
                masked.append("[MASK]")
            else:
                assert False
    else:
        assert False
    if prefix_tokens is not None:
        return prefix_tokens + masked, prefix_tokens + non_masked
    else:
        return masked, non_masked


def remove_prefix(inputs, prefix):
    return [i[len(p):] for (i, p) in zip(inputs, prefix)]


def apply_compression(sentences, labels, add_prefix=True, k=10):
    comp_masks, comp_nomasks = [], []
    for sent, label in zip(sentences, labels):
        comp_mask, comp_nomask = \
            convert_to_masked_input(sent, label,
                                    comp_or_recon="comp",
                                    prefix_tokens=(sent if add_prefix else None))
        comp_masks.append(comp_mask)
        comp_nomasks.append(comp_nomask)

    comp_sents, topk_pred = iterative_mask_prediction(comp_masks, comp_nomasks, k)

    if add_prefix:
        comp_sents = remove_prefix(comp_sents, sentences)
        topk_pred = remove_prefix(topk_pred, sentences)
        assert len(comp_sents) == len(topk_pred)
    return comp_sents, topk_pred


def apply_reconstruction(sentences, labels, comp_sents, add_prefix=True, k=10):
    recon_masks = []
    for sent, label, comp in zip(sentences, labels, comp_sents):
        recon_mask, _ = \
            convert_to_masked_input(sent, label,
                                    comp_or_recon="recon",
                                    prefix_tokens=(comp if add_prefix else None))
        recon_masks.append(recon_mask)

    recon_sents, topk_pred = iterative_mask_prediction(recon_masks, None, k=k)

    if add_prefix:
        recon_sents = remove_prefix(recon_sents, comp_sents)
        topk_pred = remove_prefix(topk_pred, comp_sents)
        assert len(recon_sents) == len(topk_pred)
    return recon_sents, topk_pred


def apply_compression_and_reconstruction(sentences,
                                         labels,
                                         add_prefix=True,
                                         k=10):
    comp_sents, comp_topk_pred = apply_compression(sentences,
                                                   labels,
                                                   add_prefix,
                                                   k)

    recon_sents, recon_topk_pred = apply_reconstruction(sentences,
                                                        labels,
                                                        comp_sents,
                                                        add_prefix,
                                                        k)

    return comp_sents, comp_topk_pred, recon_sents, recon_topk_pred


def mrf_log_prob(sentence, tokenized=False):
    if len(sentence) == 0:
        return 0
    if not tokenized:
        tar_sentence, _ = tokenize_and_put_ids([sentence])
        tar_sentence = sentence * len(sentence[0])
    else:
        tar_sentence = [sentence] * len(sentence)
    bert_input = encode_pad(tar_sentence, tokenized=True, add_cls_sep=True)
    diag_ids = bert_input[:, 1:-1].diag()
    n = bert_input.shape[0]
    bert_input[range(n), range(1, n+1)] = 103
    logits, _ = run_bert(bert_input, encoded=True)
    logits = torch.log_softmax(logits, dim=2)
    scores = torch.index_select(logits[:, 1:-1], 2, diag_ids)[:, range(n), range(n)].diag()
    return scores.mean().exp()


def predict_similarity(sents1, sents2, tokenized=True):
    if tokenized:
        sents1 = [' '.join(s) for s in sents1]
        sents2 = [' '.join(s) for s in sents2]

    scores = sim_model.predict(list(zip(sents1, sents2)))
    return scores/5  # maximum value is 5


def iterative_mask_prediction(masked_sentences, non_masked_sentences=None, k=10):
    is_compression = non_masked_sentences is not None
    with torch.no_grad():
        inputs = encode_pad(masked_sentences, tokenized=True)
        if is_compression:
            non_masked_inputs = encode_pad(non_masked_sentences, tokenized=True)
            assert inputs.shape == non_masked_inputs.shape

        topk_predictions = inputs.repeat([k, 1, 1]).transpose(0, 1).transpose(1, 2)
        is_mask = (inputs == MASK_ID).float()

        while True:
            logits, _ = run_bert(inputs, encoded=True)
            logits = torch.exp(logits)

            max_values = torch.max(logits, dim=2).values * is_mask
            target_idx = torch.argmax(max_values, dim=1).unsqueeze(1)

            target_index_scores = torch.stack([mat[idx] for (idx, mat) in zip(target_idx, logits)])

            if is_compression:
                target_index_scores = target_index_scores.squeeze(1)
                non_masked_orig_token_ids = non_masked_inputs.gather(1, target_idx)
                target_index_scores.scatter_(1, non_masked_orig_token_ids, torch.zeros_like(non_masked_orig_token_ids).float())
                target_index_scores = target_index_scores.unsqueeze(1)

            pred_token_ids = torch.argmax(target_index_scores, dim=2)

            pred_topk_token_ids = torch.topk(target_index_scores, k=k, dim=2)[1]
            for tar_i, topk_ids, mat in zip(target_idx, pred_topk_token_ids, topk_predictions):
                mat[tar_i] = topk_ids

            orig_tokens = inputs.gather(1, target_idx)
            is_mask_in_target_idxs = is_mask.gather(1, target_idx)
            non_masked_target_idxs = is_mask_in_target_idxs != 1
            pred_token_ids[non_masked_target_idxs] = orig_tokens[non_masked_target_idxs]

            is_mask.scatter_(1, target_idx, torch.zeros_like(pred_token_ids).float())
            inputs.scatter_(1, target_idx, pred_token_ids)

            if is_mask.sum() == 0:
                break

        lengths = [len(s) for s in masked_sentences]
        pred_sentences = [tokenizer.convert_ids_to_tokens(ps[1:i+1]) for (i, ps) in zip(lengths, inputs.cpu().numpy())]
        topk_predictions = [list(map(tokenizer.convert_ids_to_tokens, s))[1:i+1] for (i, s) in zip(lengths, topk_predictions.cpu().numpy())]
        return pred_sentences, topk_predictions

