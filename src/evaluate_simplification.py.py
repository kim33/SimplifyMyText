
import nltk
import textstat
from nltk.translate.bleu_score import sentence_bleu
from lens import download_model, LENS
from evaluate import load
import argparse

#lens_path = download_model("davidheineman/lens")
#lens_eval = LENS(lens_path, rescale=True)

sari = load("sari")

# Function to compute BLEU score between reference and candidate
def compute_bleu(reference_text, candidate_text):
    reference = [nltk.word_tokenize(reference_text.lower())]
    candidate = nltk.word_tokenize(candidate_text.lower())
    bleu_score = sentence_bleu(reference, candidate)
    return bleu_score

# Function to compute Flesch-Kincaid Readability and Grade level
def compute_fk_metrics(text):
    fk_grade_level = textstat.flesch_kincaid_grade(text)
    fk_reading_ease = textstat.flesch_reading_ease(text)
    return fk_grade_level, fk_reading_ease

# Read files
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def read_file_sari(file_path: str) -> list:
    """Read a text file and return a list of sentences."""
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file.readlines()]
    return lines

def calculate_sari_from_files(original_sentences: str, simplified_sentences: str, references: str) -> float:

    assert len(original_sentences) == len(simplified_sentences) == len(references), \
        "Files must have the same number of lines for comparison."

    sari_score = sari.compute(sources=original_sentences,predictions =simplified_sentences, references= references)
    
    return sari_score


def evaluate(original, simplified, reference) :

    original_text = read_file(original)
    candidate_text = read_file(simplified)
    bleu_score = compute_bleu(original_text, candidate_text)
    sim_fk_grade_level, sim_fk_reading_ease = compute_fk_metrics(candidate_text)

    #calculate SARI
    original_sentences = read_file_sari(original)
    simplified_sentences = read_file_sari(simplified)
    reference_sentences = read_file_sari(reference) 
    refs =  [[ref] for ref in reference_sentences]
    sari_score = calculate_sari_from_files(original_sentences, simplified_sentences, refs)

    #calculate LENS
#    scores = lens_eval.score(original_sentences, simplified_sentences, reference, batch_size=8, devices=[0])
#   total_score = 0
#    for score in scores:
#        total_score += score
#        avg_score = total_score / len(scores)


    print(f"BLEU Score: {bleu_score:.4f}")
    print(f"Flesch-Kincaid Simplified Grade Level: {sim_fk_grade_level:.2f}")
    print(f"Flesch Simplified Reading Ease: {sim_fk_reading_ease:.2f}")
    print(f"Average SARI Score: {sari_score}")
#    print("LENS Score : ", avg_score)


def main(args) :

    original_file = args.original_file  
    simplified_file = args.simplified_file 
    reference_file = args.ref_file
    evaluate(original_file,simplified_file, reference_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for evaluation.')
    
    parser.add_argument(
        '--simplified_file', default='./simplified_text/sample.txt', type=str,
        help='simplified text file path to evaluate'
    )
    parser.add_argument(
        '--original_file', default='./dataset/sample.txt', type=str,
        help='original text file path'
    )
    parser.add_argument(
        '--ref_file', default='./dataset/reference.txt', type=str,
        help='reference file path'
    )
    args = parser.parse_args()
    main(args)