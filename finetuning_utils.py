
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
    acc = s.accuracy_score(labels,preds)
    f1 = s.f1_score(labels,preds)
    pre = s.precision_score(labels,preds)
    rec = s.recall_score(labels,preds)
    
    return {'accuracy':acc,'f1':f1,'precision':pre,'recall':rec}

def model_init():
    """Returns an initialized model for use in a Hugging Face Trainer."""
    ## TODO: Return a pretrained RoBERTa model for sequence classification.
    ## See https://huggingface.co/transformers/model_doc/roberta.html#robertaforsequenceclassification.
    model = RobertaForSequenceClassification.from_pretrained('roberta-base')
    model = model.to('cuda')
    return model
