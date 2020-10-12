# EALM

The experimental scripts used in [our paper](https://arxiv.org/abs/2010.04379):

```
@inproceedings{kohita-etal-2020-ealm,
    title = "Q-learning with Language Model for Edit-based Unsupervised Summarization",
    author = "Ryosuke, Kohita and Akifumi, Wachi and Yang, Zhao and Ryuki Tachibana",
    booktitle = "EMNLP",
    year = "2020"
}
```

## Quick example
### Prerequest
```
pip install -r requirements.txt
```

### Training

```bash
sh run_train.sh
```

This will save a model `./model` after reaching 1000 updates.

### Predicting
```bash
sh run_predict.sh ./model data/sample.txt
```
