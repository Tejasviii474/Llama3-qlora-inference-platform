import evaluate
import numpy as np
from typing import List, Dict

class Evaluator:
    def __init__(self):
        self.rouge = evaluate.load('rouge')
        self.bleu = evaluate.load('bleu')
        self.bertscore = evaluate.load('bertscore')

    def evaluate_predictions(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Calculates ROUGE, BLEU, and BERTScore for a set of predictions and references.
        """
        results = {}
        
        # 1. ROUGE
        rouge_results = self.rouge.compute(predictions=predictions, references=references)
        results.update({
            'rouge1': rouge_results['rouge1'],
            'rouge2': rouge_results['rouge2'],
            'rougeL': rouge_results['rougeL']
        })
        
        # 2. BLEU
        bleu_results = self.bleu.compute(predictions=predictions, references=references)
        results['bleu'] = bleu_results['bleu']
        
        # 3. BERTScore
        # Note: BERTScore can be slow to compute as it requires a separate model (e.g. roberta)
        # Using distilbert for faster evaluation
        bert_results = self.bertscore.compute(predictions=predictions, references=references, lang="en", model_type="distilbert-base-uncased")
        results['bertscore_precision'] = np.mean(bert_results['precision'])
        results['bertscore_recall'] = np.mean(bert_results['recall'])
        results['bertscore_f1'] = np.mean(bert_results['f1'])
        
        return results

if __name__ == "__main__":
    evaluator = Evaluator()
    preds = ["The stock market went up today.", "Interest rates are expected to rise."]
    refs = ["The stock market increased today.", "The fed will raise interest rates."]
    metrics = evaluator.evaluate_predictions(preds, refs)
    print("Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
