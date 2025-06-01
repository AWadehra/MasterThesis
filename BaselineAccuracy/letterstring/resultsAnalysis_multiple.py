import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_combined_results(folder_list, save_as_pdf=False, output_name="combined_model_accuracy.pdf"):
    """
    Plots combined results from multiple folders with different prompt types
    """
    plt.figure(figsize=(14, 8))

    # Configuration
    LABEL_MAPPING = {
        'gptj6b': 'gptj_6b',
        'gptneox20b': 'gptneox_20b',
        'llama213b': 'llama2_13b',
        'llama270b': 'llama2_70b',
        'llama27b': 'llama2_7b'
    }

    MODELS_TO_EXCLUDE = ['gpt2', 'gptneo']
    COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    MARKERS = ['o', 's', '^', 'D', 'v', 'p', '*']  # Different markers for different prompt types

    # Create a mapping from folder names to display names
    PROMPT_MAPPING = {
        'Basic_NoPrompt_fourShot': '4-shot',
        'Basic_NoPrompt_oneShot': '1-shot',
        'Basic_NoPrompt_threeShot': '3-shot',
        'Basic_NoPrompt_twoShot': '2-shot'
    }

    for folder_idx, folder_path in enumerate(folder_list):
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        prompt_type = os.path.basename(folder_path)
        prompt_display_name = PROMPT_MAPPING.get(prompt_type, prompt_type)

        for csv_file in csv_files:
            file_path = os.path.join(folder_path, csv_file)
            df = pd.read_csv(file_path)
            original_name = df['model_name'].iloc[0]

            if original_name in MODELS_TO_EXCLUDE:
                continue

            display_name = LABEL_MAPPING.get(original_name, original_name)

            # Create a combined label with model and prompt type
            combined_label = f"{display_name} ({prompt_display_name})"

            # Plot with consistent styling
            plt.errorbar(
                x=display_name,
                y=df['Accuracy'].iloc[0],
                yerr=[[df['Accuracy'].iloc[0] - df['CI_low'].iloc[0]],
                      [df['CI_high'].iloc[0] - df['Accuracy'].iloc[0]]],
                fmt=MARKERS[folder_idx % len(MARKERS)],
                capsize=5,
                markersize=10,
                linewidth=2,
                color=COLORS[folder_idx % len(COLORS)],
                label=combined_label,
                alpha=0.8
            )

    # Enhanced styling
    plt.title('Model Accuracy Comparison Across Prompt Types', pad=20)
    plt.xlabel('Model', labelpad=10)
    plt.ylabel('Accuracy', labelpad=10)
    plt.xticks(rotation=45, ha='right')

    # Legend with model size indicators
    plt.legend(
        title='Models and Prompt Types',
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.
    )

    # Grid and layout
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.gca().set_axisbelow(True)
    plt.tight_layout()

    # Adjust y-axis limits if needed
    plt.ylim(0.7, 1)  # Adjust based on your data range

    if save_as_pdf:
        plt.savefig(output_name, format='pdf', bbox_inches='tight', dpi=300)
        print(f"Graph saved as {output_name}")

    plt.show()

# List of folders to process
folder_list = [
    'results/Basic_NoPrompt_fourShot',
    'results/Basic_NoPrompt_oneShot',
    'results/Basic_NoPrompt_threeShot',
    'results/Basic_NoPrompt_twoShot'
]

# Run the function
plot_combined_results(folder_list, save_as_pdf=True, output_name="results/combined_prompt_results.pdf")