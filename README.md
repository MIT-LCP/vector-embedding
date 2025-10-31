# Vector Embedding Project for Medical Data


## Overview

This repository provides comprehensive pipelines for generating vector embeddings from MIMIC III and MIMIC IV datasets, as well as related healthcare datasets. The primary focus is on creating high-quality vector representations that can be used for downstream machine learning tasks in healthcare AI research.

<img src="img/overview.jpg" alt="Overview" width="600"/>

## Key Features

- **Multi-dataset Support**: Full compatibility with MIMIC III, MIMIC IV, and related healthcare datasets
- **State-of-the-art Models**: Integration with advanced embedding models.

## Data Access and Setup

### MIMIC Dataset Access

1. **Register for PhysioNet Access**:
   - Create an account at [PhysioNet](https://physionet.org/)
   - Complete the required training for human subjects research
   - Request access to MIMIC III and/or MIMIC IV datasets

2. **Download Datasets**:
   ```bash
   # Example for MIMIC IV version 3.1
   wget -r -N -c -np --user YOUR_USERNAME --ask-password \
     https://physionet.org/files/mimiciv/3.1/
   ```

## CXR Model Information
### CheXagent
- **Repository**: [https://github.com/Stanford-AIMI/CheXagent](https://github.com/Stanford-AIMI/CheXagent)
- **Hugging Face**: [https://huggingface.co/StanfordAIMI/CheXagent-8b](https://huggingface.co/StanfordAIMI/CheXagent-8b)
- **Citation**: Chen, Z., et al. (2023). "CheXagent: Towards a Foundation Model for Chest X-Ray Interpretation" arXiv preprint arXiv:2401.12208

### CheXFound
- **Repository**: [https://github.com/RPIDIAL/CheXFound](https://github.com/RPIDIAL/CheXFound)
- **Citation**: Yang, Z., Xu, X., Zhang, J., Wang, G., Kalra, M. K., & Yan, P. (2025). Chest X-ray Foundation Model with Global and Local Representations Integration. ArXiv. https://arxiv.org/abs/2502.05142

### EVA-X
- **Repository**: [https://github.com/hustvl/EVA-X](https://github.com/hustvl/EVA-X)
- **Hugging Face**: [https://huggingface.co/MapleF/eva_x/tree/main](https://huggingface.co/MapleF/eva_x/tree/main)
- **Citation**: Yao, J., Wang, X., Song, Y., Zhao, H., Ma, J., Chen, Y., Liu, W., & Wang, B. (2024). EVA-X: A foundation model for general chest X-ray analysis with self-supervised learning.

### CXR Foundation Model (ELIXR)
- **Documentation**: [https://developers.google.com/health-ai-developer-foundations/cxr-foundation](https://developers.google.com/health-ai-developer-foundations/cxr-foundation)
- **Repository**: [https://github.com/google-health/cxr-foundation](https://github.com/google-health/cxr-foundation)
- **Hugging Face**: [https://huggingface.co/google/cxr-foundation](https://huggingface.co/google/cxr-foundation)

### MedSigLIP
- **Repository**: [https://github.com/google-health/medsiglip](https://github.com/google-health/medsiglip)
- **Hugging Face**: [https://huggingface.co/google/medsiglip-448](https://huggingface.co/google/medsiglip-448)
- **Documentation**: [https://developers.google.com/health-ai-developer-foundations/medsiglip](https://developers.google.com/health-ai-developer-foundations/medsiglip)
- **Citation**: Google Health AI. (2024). MedSigLIP: Medical image understanding with SigLIP.

### TorchXRayVision
- **Repository**: [https://github.com/mlmed/torchxrayvision](https://github.com/mlmed/torchxrayvision)
- **Documentation**: [https://mlmed.org/torchxrayvision/](https://mlmed.org/torchxrayvision/)
- **Citation**: Cohen, J. P., Viviano, J. D., Bertin, P., Morrison, P., Torabian, P., Guarrera, M., Lungren, M. P., Chaudhari, A., Brooks, R., Hashir, M., & Bertrand, H. (2022). TorchXRayVision: A library of chest X-ray datasets and models. Proceedings of Machine Learning for Health, 172, 231-249.


## ECHO Model Information
### EchoPrime
- **Repository**: [https://github.com/echonet/EchoPrime](https://github.com/echonet/EchoPrime)
### R3D-Transformer
- **Repository**: [https://github.com/Team-Echo-MIT/r3d-v0-embeddings](https://github.com/Team-Echo-MIT/r3d-v0-embeddings)
### PanEcho
- **Repository**: [https://github.com/CarDS-Yale/PanEcho](https://github.com/CarDS-Yale/PanEcho)

## ECG Model Information
### HuBERT-ECG
- **Repository**: [https://github.com/Edoar-do/HuBERT-ECG?tab=readme-ov-file](https://github.com/Edoar-do/HuBERT-ECG?tab=readme-ov-file)
- **Hugging Face**: [https://huggingface.co/Edoardo-BS](https://huggingface.co/Edoardo-BS)
### ECGFM-KED
- **Repository**: [https://github.com/control-spiderman/ECGFM-KED](https://github.com/control-spiderman/ECGFM-KED)
### ECGFounder
- **Repository**: [https://github.com/PKUDigitalHealth/ECGFounder](https://github.com/PKUDigitalHealth/ECGFounder)
- **Hugging Face**: [https://huggingface.co/PKUDigitalHealth/ECGFounder/tree/main](https://huggingface.co/PKUDigitalHealth/ECGFounder/tree/main)

## PPG Model Information
### PaPaGei
- **Repository**: [https://github.com/Nokia-Bell-Labs/papagei-foundation-model](https://github.com/Nokia-Bell-Labs/papagei-foundation-model)



## Related Publications

- Johnson, A. E. W., et al. "MIMIC-IV, a freely accessible electronic health record dataset." Scientific Data 10.1 (2023): 1.
- Goldberger, A. L., et al. "PhysioBank, PhysioToolkit, and PhysioNet: components of a new research resource for complex physiologic signals." Circulation 101.23 (2000): e215-e220.
- Tohyama T, et al. Multi-view echocardiographic embedding for accessible AI development. medRxiv. 2025. doi:10.1101/2025.08.15.25333725.
- Chung DJ, et al. Echocardiogram Vector Embeddings Via R3D Transformer for the Advancement of Automated Echocardiography. JACC Adv 2024;3:101196.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Embedding Library Availability
- Embedding of the MIMIC chest-x-ray, echo, ecg will be made
available on physionet. 
- Due to process time, this is currently temporarily available at 
https://drive.google.com/drive/folders/18pOkTlu-LguUZp3TLKqZSjSBky-alhja?usp=sharing

## Support and Contact

- **Issues**: [GitHub Issues](https://github.com/MIT-LCP/vector-embedding/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MIT-LCP/vector-embedding/discussions)

## Organizations

| [<img src="img/physionet-logo.jpg" alt="PhysioNet" width="200"/>](https://physionet.org/) | [<img src="img/mit-critical-data-logo.jpg" alt="MIT Critical Data" width="150"/>](https://criticaldata.mit.edu/)|
|:---:|:---:|


</div>

## Acknowledgments
- **Funding**: This research was supported by a grant of the Korea Health Technology R&D Project through the Korea Health Industry Development Institute (KHIDI), funded by the Ministry of Health & Welfare, Republic of Korea (grant number: RS-2024-00439677)

---

**Disclaimer**: This software is provided for research purposes only. It is not intended for clinical use. Always comply with your institution's ethics and data usage policies when working with healthcare data.