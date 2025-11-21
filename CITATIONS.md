# Dataset Citations

This project trains on **RefCOCO**, a referring-expression benchmark built on MS COCO images. Please cite both the referring-expression papers and MS COCO when using this code or dataset artifacts.

## RefCOCO / ReferItGame

- *ReferItGame: Referring to Objects in Photographs of Natural Scenes* (EMNLP 2014) introduced the large-scale annotation effort for referring expressions.
- *Modeling Context in Referring Expressions* (ECCV 2016) released the RefCOCO/RefCOCO+/RefCOCOg splits widely used today.

```bibtex
@inproceedings{kazemzadeh-etal-2014-referitgame,
  title        = {ReferItGame: Referring to Objects in Photographs of Natural Scenes},
  author       = {Kazemzadeh, Sahar and Ordonez, Vicente and Matten, Mark and Berg, Tamara},
  booktitle    = {Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  editor       = {Moschitti, Alessandro and Pang, Bo and Daelemans, Walter},
  month        = oct,
  year         = {2014},
  address      = {Doha, Qatar},
  publisher    = {Association for Computational Linguistics},
  pages        = {787--798},
  url          = {https://aclanthology.org/D14-1086},
  doi          = {10.3115/v1/D14-1086}
}

@inproceedings{yu2016refcoco,
  title        = {Modeling Context in Referring Expressions},
  author       = {Yu, Licheng and Poirson, Patrick and Yang, Shan and Berg, Alexander C. and Berg, Tamara L.},
  booktitle    = {European Conference on Computer Vision (ECCV)},
  year         = {2016}
}
```

## MS COCO Images

RefCOCO expressions are grounded on MS COCO images. Cite MS COCO whenever you redistribute the data or report metrics derived from it.

```bibtex
@inproceedings{lin2014coco,
  title        = {{Microsoft COCO: Common Objects in Context}},
  author       = {Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C. Lawrence},
  booktitle    = {European Conference on Computer Vision (ECCV)},
  year         = {2014}
}
```

## Dataset Access

- Hugging Face mirror: https://huggingface.co/datasets/lmms-lab/RefCOCO  
- Original ReferItGame resources: http://tamaraberg.com/research/RefExp/

When publishing results, cite the papers above and describe any modifications (e.g., multi-phrase expansion) you applied to the dataset. 
