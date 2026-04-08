"""
Compare performance of different models on the API prediction task.
"""

import json
import argparse
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np


def load_results(file_path: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(file_path) as f:
        return json.load(f)


def compare_models(result_files: List[str], model_names: List[str] = None) -> None:
    """
    Compare multiple model results and generate comparison report.
    
    Args:
        result_files: List of paths to result JSON files
        model_names: Optional list of display names for models
    """
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(result_files))]
    
    if len(model_names) != len(result_files):
        raise ValueError("Number of model names must match number of result files")
    
    # Load all results
    all_results = []
    for file_path, name in zip(result_files, model_names):
        results = load_results(file_path)
        results['display_name'] = name
        all_results.append(results)
    
    # Print comparison table
    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print()
    
    # Overall metrics
    print(f"{'Model':<20} {'Completion %':<15} {'Avg Grade':<12} {'Params':>10}")
    print("-" * 80)
    
    for r in all_results:
        model_name = r['display_name']
        completion = r['overall_completion_rate'] * 100
        grade = r['overall_grade']
        model_id = r.get('model', 'N/A')
        
        print(f"{model_name:<20} {completion:>6.1f}%        {grade:>6.3f}      {model_id:>10}")
    
    print()
    
    # By difficulty
    print("=" * 80)
    print("PERFORMANCE BY DIFFICULTY")
    print("=" * 80)
    
    difficulties = ['easy', 'medium', 'hard']
    
    for diff in difficulties:
        print(f"\n{diff.upper()}:")
        print(f"{'Model':<20} {'Completion %':<15} {'Avg Grade':<12} {'Avg Steps':<12}")
        print("-" * 80)
        
        for r in all_results:
            if diff in r.get('stats_by_difficulty', {}):
                stats = r['stats_by_difficulty'][diff]
                model_name = r['display_name']
                completion = stats['completion_rate'] * 100
                grade = stats['avg_grade']
                steps = stats['avg_steps']
                
                print(f"{model_name:<20} {completion:>6.1f}%        {grade:>6.3f}      {steps:>6.1f}")
    
    print()
    
    # Efficiency comparison (if baseline provided)
    if len(all_results) > 1:
        baseline = all_results[0]  # First model is baseline
        print("=" * 80)
        print(f"COMPARISON TO BASELINE ({baseline['display_name']})")
        print("=" * 80)
        print()
        print(f"{'Model':<20} {'Relative Grade':<18} {'Relative Completion':<20}")
        print("-" * 80)
        
        baseline_grade = baseline['overall_grade']
        baseline_completion = baseline['overall_completion_rate']
        
        for r in all_results[1:]:
            model_name = r['display_name']
            grade_ratio = (r['overall_grade'] / baseline_grade) * 100 if baseline_grade > 0 else 0
            completion_ratio = (r['overall_completion_rate'] / baseline_completion) * 100 if baseline_completion > 0 else 0
            
            print(f"{model_name:<20} {grade_ratio:>6.1f}%           {completion_ratio:>6.1f}%")
        
        print()
    
    # Generate plots
    generate_plots(all_results, difficulties)


def generate_plots(results: List[Dict], difficulties: List[str]) -> None:
    """Generate comparison plots."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
        
        model_names = [r['display_name'] for r in results]
        
        # Plot 1: Overall Completion Rate
        ax = axes[0, 0]
        completion_rates = [r['overall_completion_rate'] * 100 for r in results]
        bars = ax.bar(model_names, completion_rates, color='steelblue')
        ax.set_ylabel('Completion Rate (%)')
        ax.set_title('Overall Task Completion Rate')
        ax.set_ylim([0, 100])
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom')
        
        # Plot 2: Overall Average Grade
        ax = axes[0, 1]
        grades = [r['overall_grade'] for r in results]
        bars = ax.bar(model_names, grades, color='coral')
        ax.set_ylabel('Average Grade')
        ax.set_title('Overall Average Grade')
        ax.set_ylim([0, 1.0])
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom')
        
        # Plot 3: Completion by Difficulty
        ax = axes[1, 0]
        x = np.arange(len(difficulties))
        width = 0.8 / len(results)
        
        for i, r in enumerate(results):
            completion_by_diff = []
            for diff in difficulties:
                if diff in r.get('stats_by_difficulty', {}):
                    completion_by_diff.append(r['stats_by_difficulty'][diff]['completion_rate'] * 100)
                else:
                    completion_by_diff.append(0)
            ax.bar(x + i * width, completion_by_diff, width, label=r['display_name'])
        
        ax.set_ylabel('Completion Rate (%)')
        ax.set_title('Completion Rate by Difficulty')
        ax.set_xticks(x + width * (len(results) - 1) / 2)
        ax.set_xticklabels([d.capitalize() for d in difficulties])
        ax.legend()
        ax.set_ylim([0, 100])
        
        # Plot 4: Grade by Difficulty
        ax = axes[1, 1]
        
        for i, r in enumerate(results):
            grades_by_diff = []
            for diff in difficulties:
                if diff in r.get('stats_by_difficulty', {}):
                    grades_by_diff.append(r['stats_by_difficulty'][diff]['avg_grade'])
                else:
                    grades_by_diff.append(0)
            ax.bar(x + i * width, grades_by_diff, width, label=r['display_name'])
        
        ax.set_ylabel('Average Grade')
        ax.set_title('Average Grade by Difficulty')
        ax.set_xticks(x + width * (len(results) - 1) / 2)
        ax.set_xticklabels([d.capitalize() for d in difficulties])
        ax.legend()
        ax.set_ylim([0, 1.0])
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Plots saved to: model_comparison.png")
        
    except Exception as e:
        print(f"Could not generate plots: {e}")
        print("(matplotlib may not be available)")


def main():
    parser = argparse.ArgumentParser(description="Compare model performance")
    parser.add_argument(
        "results",
        nargs="+",
        help="Result JSON files to compare"
    )
    parser.add_argument(
        "--names",
        nargs="+",
        default=None,
        help="Display names for models (optional)"
    )
    
    args = parser.parse_args()
    
    compare_models(args.results, args.names)


if __name__ == "__main__":
    main()
