# Downloaded Datasets

This directory contains datasets for the research project. Data files are NOT
committed to git due to size. Follow the download instructions below.

## Dataset 1: AG News

### Overview
- **Source**: HuggingFace `ag_news`
- **Size**: 127,600 samples total (120K train, 7.6K test), ~30MB on disk
- **Format**: HuggingFace Dataset (Arrow format)
- **Task**: Text classification (4 classes: World, Sports, Business, Sci/Tech)
- **Splits**: train (120,000), test (7,600)
- **License**: Apache 2.0

### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset
dataset = load_dataset("ag_news")
dataset.save_to_disk("datasets/ag_news")
```

### Loading the Dataset
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/ag_news")
print(dataset['train'][0])
# {'text': '...article text...', 'label': 0}
# Labels: 0=World, 1=Sports, 2=Business, 3=Sci/Tech
```

### Sample Data
```json
[
  {"text": "Wall St. Bears Claw Back Into the Black (Reuters)...", "label": 2},
  {"text": "Carlyle Looks Toward Commercial Aerospace...", "label": 2},
  {"text": "Oil and Economy Cloud Stocks' Outlook...", "label": 2}
]
```

### Notes
- Used by Khaki et al. (2023) for drift detection with MiniLM embeddings
- World vs Sports classes provide clear semantic separation for drift experiments
- Category proportion shift is a natural way to simulate drift
- Paraphrasing within categories simulates style drift without topic shift

---

## Dataset 2: 20 Newsgroups

### Overview
- **Source**: scikit-learn `fetch_20newsgroups`
- **Size**: 18,846 samples total (11,314 train, 7,532 test), ~15MB as JSON
- **Format**: JSON files (train.json, test.json)
- **Task**: Text classification (20 newsgroup categories)
- **Splits**: train (11,314), test (7,532)
- **License**: Public domain

### Download Instructions

**Using scikit-learn:**
```python
from sklearn.datasets import fetch_20newsgroups
import json

for split in ['train', 'test']:
    data = fetch_20newsgroups(subset=split, remove=('headers', 'footers', 'quotes'))
    records = [{'text': t, 'label': int(l), 'category': data.target_names[l]}
               for t, l in zip(data.data, data.target)]
    with open(f'datasets/20newsgroups/{split}.json', 'w') as f:
        json.dump(records, f)
```

### Loading the Dataset
```python
import json
with open('datasets/20newsgroups/train.json') as f:
    data = json.load(f)
print(data[0]['category'], data[0]['text'][:100])
```

### Sample Data
```json
[
  {"text": "I was wondering if anyone out there...", "label": 7, "category": "rec.autos"},
  {"text": "A strstrly stranded strangle of str...", "label": 4, "category": "comp.sys.mac.hardware"}
]
```

### Notes
- 20 diverse categories spanning computers, recreation, science, politics, religion
- Headers, footers, and quotes removed to focus on body text
- More categories than AG News enables finer-grained drift experiments
- Useful for robustness validation across different text domains
- Categories grouped into 6 top-level groups for coarser drift simulation
