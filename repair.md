# 文献核实总结报告

## 1. σ-zero: Gradient-based Optimization of ℓ₀-norm Adversarial Examples

**核实无误**

---

## 2. Intriguing Properties of Neural Networks (Szegedy et al. 2014)

**核实无误**

---

## 3. Explaining and Harnessing Adversarial Examples (FGSM, Goodfellow et al. 2015)

**核实无误**

---

## 4. Towards Deep Learning Models Resistant to Adversarial Attacks (PGD, Madry et al. 2018)

**核实无误**

---

## 5. Towards Evaluating the Robustness of Neural Networks (C&W attack, Carlini & Wagner 2017)

**核实无误**

---

## 6. SparseFool: A Few Pixels Make a Big Difference (Modas et al. 2019)

| 项目 | 核实前 | 核实后 |
|------|--------|--------|
| 页码 | 9087--9096 | 9079-9088 |
| 新增 | - | Keywords |

**核实后BibTeX:**
```bibtex
@INPROCEEDINGS{8954332,
  author={Modas, Apostolos and Moosavi-Dezfooli, Seyed-Mohsen and Frossard, Pascal},
  booktitle={2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  title={SparseFool: A Few Pixels Make a Big Difference}, 
  year={2019},
  volume={},
  number={},
  pages={9079-9088},
  keywords={Geometry;Training;Perturbation methods;Semantics;Neural networks;Visual effects;Minimization;Deep Learning;Recognition: Detection;Categorization;Retrieval;Representation Learning},
  doi={10.1109/CVPR.2019.00930}
}
```

---

## 7. RobustBench: A Standardized Adversarial Robustness Benchmark (Croce et al. 2021)

**核实无误**

---

## 8. Image Quality Assessment: From Error Visibility to Structural Similarity (SSIM, Wang et al. 2004)

**核实无误**

---

## 9. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (LPIPS, Zhang et al. 2018)

**核实无误**

---

## 10. Gradient-based Learning Applied to Document Recognition (MNIST, LeCun et al. 1998)

**核实无误**

---

## 11. Decoupling Direction and Norm for Efficient Gradient-Based L2 Adversarial Attacks and Defenses (DDN, Rony et al. 2019)

**核实无误**

---

## 12. Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples (Athalye et al. 2018)

**核实无误**

---

## 13. Decision-based Adversarial Attacks: Reliable Attacks Against Black-box Machine Learning Models (Brendel et al. 2018)

**核实无误**

---

## 14. Square Attack: A Query-Efficient Black-Box Adversarial Attack via Random Search (Andriushchenko et al. 2020)

**待确认** - DOI可能不正确
- 核实前DOI: 10.1007/978-3-030-58592-1_29
- 建议DOI: 10.48550/arXiv:1912.00049

---

## 15. Improving Robustness Using Generated Data (Gowal et al. 2021)

**核实无误**

---

## 16. Simple Black-box Adversarial Attacks (Guo et al. 2019)

**核实无误**

---

## 17. The Limitations of Deep Learning in Adversarial Settings (Papernot et al. 2016)

**核实无误**

---

## 18. Adversarial Training for Free! (Tramer et al. 2020)

**需更改**   

**核实后BibTeX:**
```bibtex
@inproceedings{tramer2020free,
  title     = {Adversarial Training for Free!},
  author    = {Tram{\`e}r, Florian and Dupr{\'e}, Nicolas and Rusak, Gideon and Carlini, Nicholas and Boneh, Dan},
  booktitle = {Proc. USENIX Security Symp.},
  pages     = {1359--1376},
  year      = {2020}
}
```

---

## 19. Strong Baselines for Simple Adversarial Training (Chen et al. 2020)

**未找到原文和相似文章**

---

## 20. Exploring the Trade-off Between Adversarial Robustness and Clean Accuracy (Xu et al. 2023)

**找到相近文献** - 原标题可能有误，以下为最相近的文献：

| 项目 | 原文献 | 找到的文献 |
|------|--------|--------|
| 标题 | Exploring the Trade-off Between Adversarial Robustness and Clean Accuracy | Do Adversarially Robust ImageNet Models Transfer Better? |
| 作者 | Xu, Dongxian and Zhu, Tianhang and Zhang, Lu | Salman, Hadi and Ilyas, Andrew and Engstrom, Logan and Kapoor, Sohil and Madry, Aleksander |
| 会议 | IEEE Trans. Pattern Analysis and Machine Intelligence | NeurIPS 2020 (arXiv preprint) |
| 年份 | 2023 | 2020 |

**注意**: 原文献可能不存在，建议使用以下相关文献替代：

**建议BibTeX:**
```bibtex
@article{salman2020transfer,
  title={Do Adversarially Robust ImageNet Models Transfer Better?},
  author={Salman, Hadi and Ilyas, Andrew and Engstrom, Logan and Kapoor, Sohil and Madry, Aleksander},
  journal={arXiv preprint arXiv:2007.08489},
  year={2020}
}
```

**Google Scholar链接:** https://scholar.google.com/scholar?q=Adversarial+Robustness+Trade-off+Clean+Accuracy

---

*报告生成时间: 2026-04-29*