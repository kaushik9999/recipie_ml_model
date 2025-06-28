from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
import re

def preprocess_text(text):
    """
    Preprocess text for metric calculation to handle recipe-specific formatting.
    
    Args:
        text (str): Raw text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    # Convert to lowercase for better matching
    text = text.lower()
    
    # Remove bullet points, numbering, and other recipe-specific formatting
    text = re.sub(r'•|\d+\.', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def compute_metrics(predictions, references):
    """
    Compute BLEU and ROUGE scores for generated recipes.
    
    Args:
        predictions (list): List of generated recipe strings.
        references (list): List of ground truth recipe strings.
    
    Returns:
        dict: Dictionary with BLEU and ROUGE scores.
    """
    # Preprocess texts
    processed_preds = [preprocess_text(pred) for pred in predictions]
    processed_refs = [preprocess_text(ref) for ref in references]
    
    # Compute BLEU score
    bleu_score = corpus_bleu(processed_preds, [processed_refs]).score

    # Compute ROUGE score (ROUGE-L is suitable for longer text like recipes)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Calculate all ROUGE metrics
    rouge_scores = {}
    for key in ['rouge1', 'rouge2', 'rougeL']:
        scores = [scorer.score(ref, pred)[key].fmeasure for ref, pred in zip(processed_refs, processed_preds)]
        rouge_scores[f"ROUGE-{key[-1].upper()}"] = sum(scores) / len(scores) if scores else 0

    # Combine metrics
    metrics = {
        "BLEU": bleu_score,
        **rouge_scores
    }
    
    return metrics

if __name__ == "__main__":
    # Example usage for testing with recipe-like data
    preds = [
        "Ingredients: • 2 cups chicken • 1 cup rice\n\nDirections:\n1. boil water\n2. cook rice\n3. add chicken"
    ]
    refs = [
        "Ingredients: • 2 cups chicken • 1 cup rice • salt\n\nDirections:\n1. boil water\n2. cook rice with salt\n3. add chicken"
    ]
    
    metrics = compute_metrics(preds, refs)
    print("Metrics on sample data:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")