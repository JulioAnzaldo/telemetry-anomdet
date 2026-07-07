# Citations

`telemetry-anomdet` builds on prior research in anomaly detection, explainability, and
symbolic distillation. This file credits the works whose methods the toolkit implements
or draws on. To cite the toolkit **itself**, see [CITATION.cff](CITATION.cff).

Tags: *(implemented)*: the method is built into the toolkit; *(planned)*: scheduled on
the roadmap; *(reference)*: informs the design or serves as a benchmark.

## Detection methods

- **telemanom** *(reference / benchmark)*: LSTM forecasting with nonparametric dynamic
  thresholding; also the origin of the SMAP/MSL benchmark used here.
  Hundman, K., Constantinou, V., Laporte, C., Colwell, I., & Soderstrom, T. (2018).
  *Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding.*
  KDD '18, 387–395. https://doi.org/10.1145/3219819.3219845

- **GDN: Graph Deviation Network** *(planned for Phase 2)*: learns inter-sensor
  relationships and scores relational deviations invisible to univariate methods.
  Deng, A., & Hooi, B. (2021). *Graph Neural Network-Based Anomaly Detection in
  Multivariate Time Series.* AAAI 2021. https://arxiv.org/abs/2106.06947

- **TranAD** *(planned for Phase 2)*: dual-phase transformer reconstruction over raw windows.
  Tuli, S., Casale, G., & Jennings, N. R. (2022). *TranAD: Deep Transformer Networks for
  Anomaly Detection in Multivariate Time Series Data.* https://arxiv.org/abs/2201.07284

- **MEMTO** *(reference / SOTA target)*: memory-guided transformer reconstruction.
  Song, J., Kim, K., Oh, J., & Cho, S. (2023). *MEMTO: Memory-guided Transformer for
  Multivariate Time Series Anomaly Detection.* https://arxiv.org/abs/2312.02530

- **Temporal Attention LSTM Autoencoder** *(reference)*
  Xu, Z., Cheng, Z., & Guo, B. (2023). *A Multivariate Anomaly Detector for Satellite
  Telemetry Data Using Temporal Attention-Based LSTM Autoencoder.* IEEE Transactions on
  Instrumentation and Measurement, 72. https://doi.org/10.1109/TIM.2023.3296125

## Explainability

- **SHAP** *(planned for Phase 3)*: per-channel attribution over ensemble score components.
  Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model
  Predictions.* NeurIPS. https://arxiv.org/abs/1705.07874

- **Survey on Explainable Anomaly Detection** *(reference)*
  Li, Z., Zhu, Y., & van Leeuwen, M. (2022). *A Survey on Explainable Anomaly Detection.*
  https://arxiv.org/abs/2210.06959

## Symbolic distillation (edge deployment)

- **SymTorch** *(stretch goal)*: distills trained networks into closed-form expressions
  via symbolic regression for microcontroller execution.
  Tan, E. S., Soubki, A., & Cranmer, M. D. (2026). *SymTorch: A Framework for Symbolic
  Distillation of Deep Neural Networks.* https://arxiv.org/abs/2602.21307

## LLM reasoning

- **Can LLMs Understand Time Series Anomalies?** *(reference for Phase 4)* — motivates
  supplying SHAP charts to the LLM as images rather than token sequences.
  Zhou, Z., & Yu, R. (2025). *Can LLMs Understand Time Series Anomalies?* ICLR 2025.
  https://arxiv.org/abs/2410.05440

## Datasets

- **SMAP: Soil Moisture Active Passive** *(primary)*: NASA spacecraft telemetry,
  released with Hundman et al. (2018).
- **OPS-SAT** *(cross-dataset generalization)*: ESA mission telemetry.

## Surveys & standards

- Fejjari, A., Delavault, A., Camilleri, R., & Valentino, G. (2025). *A Review of Anomaly
  Detection in Spacecraft Telemetry Data.* Applied Sciences, 15(10), 5653.
  https://doi.org/10.3390/app15105653
- National Institute of Standards and Technology (2023). *Artificial Intelligence Risk
  Management Framework (AI RMF 1.0).* NIST AI 100-1.
  https://www.nist.gov/itl/ai-risk-management-framework
