import json
import glob
import os
HOME = os.getenv("HOME")

def analyze_ratios(summary_dir):
    """Calculate average proportion of correct answers across all questions."""
    summary_files = glob.glob(os.path.join(summary_dir, "question_*_summary.json"))
    proportions = []
    
    for file_path in summary_files:
        with open(file_path, 'r') as f:
            summary = json.load(f)
        
        if 'individual_answers_final' in summary:
            individual_answers = summary['individual_answers_final']
        else:
            individual_answers = summary['individual_answers']
        
        if 'correct_answer_reference' in summary:
            correct_answer = summary['correct_answer_reference']
        else:
            correct_answer = summary['correct_answer_letter']
        
        correct_count = sum(1 for answer in individual_answers if answer == correct_answer)
        total_count = len(individual_answers)
        
        proportion = correct_count / total_count if total_count > 0 else 0
        proportions.append(proportion)
    
    return sum(proportions) / len(proportions) if proportions else 0

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python analyze_ratio_correct_vs_incorrect.py <model_name> <dataset_name> <run_name>")
        sys.exit(1)
    
    model_name = sys.argv[1]
    dataset_name = sys.argv[2]
    run_name = sys.argv[3]

    summary_dir = f"{HOME}/slimsc/prune/results/{model_name}/{dataset_name}/{run_name}/summaries"

    avg_proportion = analyze_ratios(summary_dir)
    print(f"Average proportion of correct answers: {avg_proportion:.3f}")

    output_dir = f"{HOME}/slimsc/prune/analysis/{model_name}/{dataset_name}"

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, f"ratio_analysis_{run_name}.txt"), "w") as out_f:
        out_f.write(f"Avg. proportion of final chains that got the correct answer: {avg_proportion:.3f}") 
    
    