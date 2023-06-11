import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from transformers import create_optimizer
from sklearn.metrics import accuracy_score, f1_score
from scipy.special import softmax
from scipy.stats import entropy
from scipy.spatial import distance
from sklearn.linear_model import LinearRegression
import sys


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


# given 2d array of probabilities over each class (row is a single data point, ncols= #classes)
# returns a single score representing the vector
def score(probs, method='max'):
    if method == 'max':
        return np.max(probs, axis=1)
    elif method == 'ne':
        return -1 * entropy(probs, axis=1)
    elif method == 'l2n':
        return np.linalg.norm(probs, axis=1)

    n_classes = probs.shape[1]
    unif = np.repeat(1 / n_classes, n_classes).reshape((1, -1))
    if method == 'l1':
        return distance.cdist(probs, unif, metric='cityblock')
    elif method == 'l2':
        return distance.cdist(probs, unif, metric='euclidean')
    elif method == 'js':
        return distance.cdist(probs, unif, metric='jensenshannon')


def predict_target_metric_atc(source_metric, source_prob, target_prob, score_method='max'):
    source_scores = score(source_prob, score_method)
    target_scores = score(target_prob, score_method)
    thresholds = np.unique(source_scores)
    threshold_curve = np.array(
        [[t, abs(np.count_nonzero(source_scores < t) / len(source_scores) - source_metric)] for t in thresholds])
    optimal_t = threshold_curve[np.argmin(threshold_curve[:, 1]), 0]
    target_metric = np.count_nonzero(target_scores < optimal_t) / len(target_scores)

    return target_metric


def doc(source_probs, target_probs):
    source_conf = np.mean(np.max(source_probs, axis=1))
    target_conf = np.mean(np.max(target_probs, axis=1))

    return source_conf - target_conf


def predict_target_metric_doc(source_probs, source_truth, target_probs, n_samples=10):
    source_truth = np.array(source_truth)

    # generate 10 bootstrap (with replacement) subsets of val set, each of size len/10
    ix = np.random.choice(source_probs.shape[0], (n_samples, int(np.ceil(source_probs.shape[0] / n_samples))))

    # fix the first subset, this is what we will compare the rest to
    comp_source_probs, comp_source_truth = source_probs[ix[0]], source_truth[ix[0]]
    comp_acc = (np.argmax(comp_source_probs, axis=1) == comp_source_truth).mean()

    dist_vs_acc = []
    for i in ix[1:]:
        curr_source_probs, curr_source_truth = source_probs[i], source_truth[i]
        dist = doc(comp_source_probs, curr_source_probs)
        acc = (np.argmax(curr_source_probs, axis=1) == curr_source_truth).mean()
        dist_vs_acc.append([dist, comp_acc - acc])  # store distance and accuracy drop

    # get regression of accuracy (on each batch) vs distance
    dist_vs_acc = np.array(dist_vs_acc)
    reg = LinearRegression().fit(dist_vs_acc[:, 0].reshape(-1, 1), dist_vs_acc[:, 1])

    # then get distance to target set (doc)
    target_dist = doc(comp_source_probs, target_probs)
    target_acc = reg.predict(np.array(target_dist).reshape(-1, 1))[0] + comp_acc

    return target_acc


def run_all_experiments(datasetname, config, dims, n_runs):
    print("--DATASET: " + datasetname)

    master_df = load_dataset(datasetname, config) if config else load_dataset(datasetname)

    # generate validation set of 20% from training if doesn't exist
    if 'validation' not in master_df:
        split = master_df['train'].train_test_split(test_size=0.2)

        master_df['train'] = split['train']
        master_df['validation'] = split['test']

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def preprocess_text(lst):
        return tokenizer(lst['text'], truncation=True, padding=True)

    master_df_tokenized = master_df.map(preprocess_text, batched=True)

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=256,
        per_device_eval_batch_size=256,
        num_train_epochs=5,
        weight_decay=0.01,
    )

    result = []

    for d in dims:
        print("Starting d=",d)

        # filter to only keep first d dimension
        df_tokenized = master_df_tokenized.filter(lambda s: s["label"] in range(d))

        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=d)

        trainer = Trainer(
            model=model,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=df_tokenized["train"],
            tokenizer=tokenizer
        )

        trainer.train()

        test_pred = trainer.predict(df_tokenized["test"])
        test_prob, test_acc = softmax(test_pred.predictions, axis=1), test_pred.metrics['test_accuracy']

        ix = np.random.choice(df_tokenized["validation"].shape[0], (n_runs, df_tokenized["validation"].shape[0]))

        for i in range(n_runs):
            print("d=", d, "run=", i)
            curr_val = df_tokenized["validation"].select(ix[i])
            val_pred = trainer.predict(curr_val)
            val_prob, val_acc = softmax(val_pred.predictions, axis=1), val_pred.metrics['test_accuracy']
            res = [predict_target_metric_atc(val_acc, val_prob, test_prob, score_method='max'),
                   predict_target_metric_atc(val_acc, val_prob, test_prob, score_method='ne'),
                   predict_target_metric_atc(val_acc, val_prob, test_prob, score_method='l2n'),
                   predict_target_metric_atc(val_acc, val_prob, test_prob, score_method='l1'),
                   predict_target_metric_atc(val_acc, val_prob, test_prob, score_method='l2'),
                   predict_target_metric_atc(val_acc, val_prob, test_prob, score_method='js'),
                   predict_target_metric_doc(val_prob, curr_val["label"], test_prob)]
            res = np.multiply(100, res) # want it in terms of percentage points
            res = np.abs(np.subtract(100*test_acc, res))
            res = np.insert(res, 0, (d, i))  # store run number and dimension
            result.append(res)

        print("Done d=", d)


    result_df = pd.DataFrame(result,
                             columns=['dim', 'run', 'atc_max', 'atc_ne', 'atc_l2n', 'atc_l1', 'atc_l2', 'atc_js',
                                      'doc'])

    result_df_long = pd.melt(result_df, id_vars=['dim', 'run'], var_name='method', value_name='error').reindex(
        columns=['dim', 'method', 'run', 'error'])

    return result_df_long


if __name__ == "__main__":
    args = sys.argv[1:]

    # get arguments
    # 1: name of data set from Huggingface
    # 2: max number of dimensions (inclusive); will test dimensions 2,3,4,...max
    # 3: number of runs per method per dimension
    dataset_name = args[0]
    filename = dataset_name + "_results.csv"
    config = None if args[1]=="none" else args[1]
    dims = list(range(2,int(args[2])+1))
    n_runs = int(args[3])

    # run the experiment
    run_all_experiments(dataset_name, config, dims, n_runs).to_csv(filename)


# RUNFILE
# emotion, none
# tweet_eval, emoji
# banking77, none