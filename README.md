# AI Text Search (aits)

# Usage
## 1. Setup Environment  
1.1 Clone the repository
`git clone https://github.com/AnkushMalaker/aits.git`  
1.2 Create conda or venv environment  
`conda create -n py39 python=3.9`  
1.3 Activate env and install dependencies  
```
conda activate py39
cd aits
pip install -r requirements.txt
```
## 2. Documents Setup
Place all the documents to search within `DocsToSearch`.

## 3. Create Index  
From the terminal run `src/scripts/create_index.py "./DocsToSearch"`  
You can specify a different directory instead of `./DocsToSearch`. This creates the index at `$HOME/.cache/aits/latest_index.pkl`. You can specify a different path with `--save_path`

## 4. Run Query
To search the documents for closest match to "Hey, this is an example sentence", use:  
`python3 src/scripts/ai_search.py --index_path "<PATH/TO/INDEX. Leave this arguemnt to use default path>" --query "Hey, this is an example sentence"`  
If you don't supply a `--query` tag, user will be asked to provide one.
You can use `-c` tag for multiple queries.
You can use `-k` or `--top_k` arguemnt to specify how many matches you want per query.
