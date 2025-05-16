#%% Load all data
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import os, json
from tqdm import tqdm
import pandas as pd

all_subjects = ['biology', 'mathematics', 'chemistry', 'physics', 
    'geoscience', 'astronomy', 'sociology', 'jurisprudence', 
    'political science', 'economics', 'psychology', 'pedagogy', 
    'civil engineering', 'mechanical engineering', 'medicine', 
    'computer science', 'agronomy', 'literature', 'history', 
    'philosophy', 'art', 'material science', 'environmental science', 
    'sports science', 'data science']
data_dir = 'data/wikidata/s15_final_cleaned_data/cleaned'
all_data = {}
for sub in tqdm(all_subjects):
    sub_path = os.path.join(data_dir, '{}.json'.format(sub))
    with open(sub_path, 'r') as f:
        all_data[sub] = json.load(f)
#%%
plt.rcParams['font.family'] = 'Times New Roman'
all_subjects = {
    'Natural Science': ['biology', 'mathematics', 'chemistry', 'physics',
        'geoscience', 'astronomy'],
    'Humanity': ['literature', 'history', 'philosophy', 'art'],
    'Social Science': ['sociology', 'jurisprudence',
        'political science', 'economics', 'psychology', 'pedagogy'],
    'Applied Science': ['civil engineering', 'mechanical engineering',
        'medicine', 'computer science', 'agronomy'],
    'Interdisciplinary Studies': ['material science', 'environmental science',
        'sports science', 'data science']
}
fig, ax = plt.subplots()
knowl_count = {sub_class: {sub:len(all_data[sub]) for sub in subs} 
               for sub_class, subs in all_subjects.items()}
knowl_count
#%%
def plot_subject_knowledge_bar_chart(subject_dict):
    # Prepare the data
    categories = list(subject_dict.keys())
    category_totals = []
    subject_names = []
    subject_knowledge = []

    for category, subjects in subject_dict.items():
        total_knowledge = sum(subjects.values())
        category_totals.append(total_knowledge)
        for subject, knowledge_count in subjects.items():
            subject_names.append(subject)
            subject_knowledge.append(knowledge_count)

    # Set up the bar chart
    bar_width = 0.5
    ind = np.arange(len(categories))
    bottom = np.zeros(len(categories))
    
    # Plot each subject in each category
    for i, category in enumerate(categories):
        subjects = subject_dict[category]
        subject_bar_positions = []
        for j, (subject, knowledge_count) in enumerate(subjects.items()):
            subject_bar_positions.append(bar_width)
            plt.bar(ind[i], knowledge_count, bar_width, label=f"{category}: {subject}" if j == 0 else "", bottom=bottom[i])
            bottom[i] += knowledge_count

    # Customizing the chart
    plt.xticks(ind, categories, rotation=45)
    plt.ylabel('Knowledge Count')
    plt.title('Knowledge Distribution by Subject Category')
    plt.legend(title="Subjects", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Show the plot
    plt.tight_layout()
    plt.show()
 
# Call the function with the example data
plot_subject_knowledge_bar_chart(knowl_count)
