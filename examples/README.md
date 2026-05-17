# Example Data

This directory contains tiny SPA-VL-derived examples for schema and metric verification.

```text
spavl_detection.csv
spavl_detection_predictions.csv
spavl_explanation_bbox.csv
spavl_explanation_bbox_predictions.csv
images/
```

The detection CSV has one safe row and one unsafe row from SPA-VL (`dataset_C`). The bbox abduction CSV uses SPA-VL rows from the bbox-abduction test split, where qualified visual evidence is available.

Run the CPU-only verification:

```bash
bash scripts/verify_examples.sh
```

This checks the detection metric parser and bbox metric parser without loading an 8B model. Full PT extraction, training, and model evaluation still require the corresponding base model, trained checkpoint, and GPU resources.

## Attribution

The example rows and images are derived from SPA-VL, which is released under CC BY 4.0 on Hugging Face: https://huggingface.co/datasets/sqrti/SPA-VL

Please cite:

```bibtex
@misc{zhang2024spavl,
  title={SPA-VL: A Comprehensive Safety Preference Alignment Dataset for Vision Language Model},
  author={Yongting Zhang and Lu Chen and Guodong Zheng and Yifeng Gao and Rui Zheng and Jinlan Fu and Zhenfei Yin and Senjie Jin and Yu Qiao and Xuanjing Huang and Feng Zhao and Tao Gui and Jing Shao},
  year={2024},
  eprint={2406.12030},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
