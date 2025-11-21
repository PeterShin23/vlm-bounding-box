# Dataset Citations

This project uses the **RefCOCO** dataset for referring expression grounding training and evaluation.

## RefCOCO Dataset

### Citation

**Paper:**
Licheng Yu, Patrick Poirson, Shan Yang, Alexander C. Berg, Tamara L. Berg. "Modeling Context in Referring Expressions", ECCV 2016.

**BibTeX:**
```bibtex
@inproceedings{yu2016refcoco,
  title={Modeling Context in Referring Expressions},
  author={Yu, Licheng and Poirson, Patrick and Yang, Shan and Berg, Alexander C. and Berg, Tamara L.},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2016}
}
```

### Dataset Information

**HuggingFace:** https://huggingface.co/datasets/lmms-lab/RefCOCO

**Original Website:** https://github.com/lichengunc/refer

**Dataset Statistics:**
- **val split**: 8,811 samples (use for **training**)
- **test split**: 5,000 samples (use for **evaluation**)
- **testA split**: 1,975 samples (people subset)
- **testB split**: 1,810 samples (objects subset)

⚠️ **IMPORTANT**: RefCOCO has NO "train" split on HuggingFace. Use **"val"** split for training!

**Data Sources:**
- Images from MS COCO dataset
- Referring expressions collected via Amazon Mechanical Turk
- Multiple expressions per object with different levels of detail

**Task:**
- Given an image and a referring expression (phrase), predict the bounding box of the described object/region
- Referring expressions: Natural language descriptions like "the red car on the left", "person wearing blue shirt"

**Variants:**
- **RefCOCO**: Original dataset (allows location words)
- **RefCOCO+**: No location words allowed in expressions
- **RefCOCOg**: Longer, more complex expressions

This project uses the **RefCOCO** variant.

### Copyright and Usage

**Copyright:** See original RefCOCO repository for licensing information.

**Usage:** When using this dataset, please cite the original ECCV 2016 paper above.

## Dataset Description

### Purpose

RefCOCO was created to advance research in referring expression grounding:
- Enable models to understand natural language descriptions of objects in images
- Provide a benchmark for phrase-conditional object localization
- Study how language can be used to disambiguate between multiple objects
- Research multimodal understanding (vision + language)

### Key Features

1. **Large Scale**: Over 19,000 total samples across all splits
2. **Natural Language**: Real referring expressions collected from human annotators
3. **Challenging**: Complex scenes with multiple similar objects requiring disambiguation
4. **Multimodal**: Requires joint understanding of visual and linguistic context
5. **Well-Established**: Widely used benchmark in vision-language research

### Recommended Usage

For this project with $10 budget constraint:

**Local Debugging (MPS - FREE):**
- Split: `val`
- max_samples: 500-1000
- Purpose: Quick iteration and debugging

**Runpod Training (GPU - $3-9):**
- Split: `val` (use for training)
- max_samples: 5000-15000 depending on budget
- Evaluation: Use `test` split for final metrics

**Split Strategy:**
- **Training**: Use `val` split (8,811 samples)
- **Evaluation**: Use `test` split (5,000 samples)
- **Optional**: Use `testA`/`testB` for detailed analysis

## Related Datasets

The RefCOCO family includes three variants:

### RefCOCO (this project)
- Allows location words in expressions ("on the left", "in the back")
- More natural referring expressions
- Easier to learn spatial relationships

### RefCOCO+
- No location words allowed
- Forces use of appearance descriptions
- More challenging for models

### RefCOCOg
- Longer, more complex expressions (average 8.4 words)
- More detailed descriptions
- Google-style referring expressions

## Attribution Requirements

When publishing work using this project and the RefCOCO dataset:

1. **Cite the RefCOCO paper** (see BibTeX above)
2. **Cite the MS COCO dataset** (images are from COCO):
   ```bibtex
   @inproceedings{lin2014coco,
     title={Microsoft COCO: Common Objects in Context},
     author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence},
     booktitle={European Conference on Computer Vision (ECCV)},
     year={2014}
   }
   ```
3. **Cite this project** (if using the code):
   ```bibtex
   @misc{qwen3-vl-refcoco,
     title={Qwen3-VL Phrase Grounding with RefCOCO},
     author={Your Name},
     year={2025},
     howpublished={\url{https://github.com/yourusername/vlm-bounding-box}}
   }
   ```

## Data Access

RefCOCO is automatically downloaded from HuggingFace when you run the data preparation scripts. No manual download required!

```bash
python scripts/prepare_data.py --split val --visualize
```

The dataset will be cached in your HuggingFace cache directory (typically `~/.cache/huggingface/datasets/`).

## Questions or Issues

For questions about the RefCOCO dataset itself, please visit:
- HuggingFace: https://huggingface.co/datasets/lmms-lab/RefCOCO
- Original repository: https://github.com/lichengunc/refer

For questions about using RefCOCO with this project, please open an issue in this repository.

---

**Last Updated:** 2025-01-20
