
import sklearn.metrics as s
from transformers import RobertaTokenizer, RobertaForSequenceClassification

def compute_metrics(eval_pred):
    """Computes accuracy, f1, precision, and recall from a 
    transformers.trainer_utils.EvalPrediction object.
    """
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)

    ## TODO: Return a dictionary containing the accuracy, f1, precision, and recall scores.
    ## You may use sklearn's precision_recall_fscore_support and accuracy_score methods.
    acc = s.accuracy_score(labels,preds,normalize = True)
    f1 = s.f1_score(labels,preds,average = 'binary')
    pre = s.precision_score(labels,preds,average = 'binary')
    rec = s.recall_score(labels,preds,average = 'binary')
    x = s.precision_recall_fscore_support(labels,preds,average = 'macro')
    print(x)
    return {'accuracy':acc,'f1':x[2],'precision':x[0],'recall':x[1]}

def model_init():
    """Returns an initialized model for use in a Hugging Face Trainer."""
    ## TODO: Return a pretrained RoBERTa model for sequence classification.
    ## See https://huggingface.co/transformers/model_doc/roberta.html#robertaforsequenceclassification.
    model = RobertaForSequenceClassification.from_pretrained('roberta-base')
    model = model.to('cuda')
    return model
