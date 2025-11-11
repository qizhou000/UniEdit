# UniEdit: A Unified Knowledge Editing Benchmark for Large Language Models
![](figs/fig_UniEdit_dataset_design.svg)
**Figure 1**: Data Composition of UniEdit.
![](figs/fig_uniedit_data_construction.svg)
**Figure 2**: Data Construction of UniEdit.

## UniEdit Construction
Please refer to the `./kits/uniedit/preprocess_pipeline` folder for the construction pipeline and toolkit of UniEdit.  
We use ElasticSearch to retrieve entities, which should be downloaded and placed in `./elasticsearch`.  
For scripts used in certain steps, please refer to `./scripts/wikidata_gen`.

## UniEdit Evaluation
**1.** Download the UniEdit data and place it in the `data` folder with the following structure:
- data
    - UniEdit
        - train
            - agronomy.json
            - art.json
            - ...
        - test
            - agronomy.json
            - art.json
            - ...

To check which criteria each sample belongs to, you can run the following Python script:
```
from dataset.llm import UniEdit

data = UniEdit('data/UniEdit/test')
print(data.get_data_by_ids([0]))
```

**2.** Download the backbones and place them in the `./models` folder. The model URLs are:

- GPT2-XL-1.5B: https://huggingface.co/openai-community/gpt2-xl  
- GPT-J-6B: https://huggingface.co/EleutherAI/gpt-j-6b  
- LLaMa-3.1-8B: https://huggingface.co/meta-llama/Llama-3.1-8B

**3.** For scripts related to editor training and testing on UniEdit, please refer to `./scripts`.

**4.** For evaluation results, please refer to `./eval_results`.

**5.** For visualization of evaluation results, refer to `./kits/uniedit_visualization`, the Figures are in `./figs`.

**We are planning to enhance and refine this repository soon.**


