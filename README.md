# Generalized Learning of Accurate Electron Correlation Energy Using Low-Dimensional Data Neural Networks
# 低维数据神经网络泛化学习精确电子相关能

**Authors:** Weibin Wu (吴炜斌), Qiujiang Liang (梁秋江), Jun Yang (杨军)**  
**Affiliation:** Department of Chemistry, The University of Hong Kong

## Abstract

The computational complexity of solving for electron correlation energy grows rapidly with the increase in the number of atoms. Direct calculation of electron correlation energy for complex molecules primarily relies on lower-level electronic structure methods.

In this project, we utilize **local electronic features** (instead of purely atomic descriptors) to construct a **Transferable Deep Neural Network (T-dNN)**. By introducing compact electronic structure correlation features derived from mean-field calculations, we are able to train on a small amount of "low-dimensional" data (small molecules) and generalize to larger systems.

This model learns and predicts MP2 and CCSD electron correlation energies at chemical accuracy. It demonstrates strong transferability to larger molecules and periodic systems, including:
*   Short-chain alkanes of varying lengths
*   Peptide chains
*   Non-covalent interactions (including biomolecules)
*   Water clusters of different sizes and morphologies

## Method Overview

Unlike mainstream approaches that define descriptors based on atomic coordinates, our approach extracts compact **local correlation features** based on the results of mean-field calculations (e.g., HF). This is crucial for the neural network to retain important electron correlation patterns.

The codebase includes:
*   **TDNN:** Implementation of the Transferable Deep Neural Network located in the [`tdnn/`](tdnn/) directory.
*   **Feature Generation:** Scripts for generating local electronic features, likely utilizing the OSV-MP2 formalism, located in [`gen_feature/`](gen_feature/).

## Performance

As shown in our results (Figure 1 in associated documentation):
1.  **Generalization:** The T-dNN model achieves low Mean Absolute Error (MAE) across various datasets (ACONF, PCONF, S66, BBI, SSI) for CCSD/6-31g*.
2.  **Scalability:** The model accurately predicts the MP2/cc-pVTZ electron correlation energy for water clusters ranging from $(H_2O)_8$ to $(H_2O)_{128}$, despite being trained on smaller subsets.

## Citation

If this code or research benefits your work, please cite the following papers associated with the methodology:

### Primary References

1.  **Yang, J.** "Title of the review/perspective regarding ML in Chem." *Wiley Interdiscip. Rev. Comput. Mol. Sci.* **2024**, *14*, e1706.
2.  **Zhou, R.; Liang, Q.; Yang, J.** "Title regarding specific methodology." *J. Chem. Theory Comput.* **2020**, *16*, 196-210.
3.  **Liang, Q.; Yang, J.** "Title regarding OSV or related ML method." *J. Chem. Theory Comput.* **2021**, *17*, 6841-6860.
4.  **Ng, W.-P.; Liang, Q.; Yang, J.** "Title regarding recent improvements." *J. Chem. Theory Comput.* **2023**, *19*, 5439-5449.
5.  **Ng, W.-P.; Zhang, Z.; Yang, J.** **2024** (to be submitted).

### Contact
For questions or correspondence, please contact: **shiwei_zhang@connect.hku.hk**
