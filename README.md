# Vector Embedding Pipelines for MIMIC III/IV Datasets


## Overview

This repository provides comprehensive pipelines for generating vector embeddings from MIMIC III and MIMIC IV datasets, as well as related healthcare datasets. The primary focus is on creating high-quality vector representations that can be used for downstream machine learning tasks in healthcare AI research.

[<img src="img/overview.jpg" alt="Overview" width="600"/>]

## Key Features

- **Multi-dataset Support**: Full compatibility with MIMIC III, MIMIC IV, and related healthcare datasets
- **State-of-the-art Models**: Integration with advanced embedding models including:
  - **EchoPrime**: Specialized for echocardiography data
  - **R3D-Transformer**: 3D ResNet-based transformer for temporal medical data
  - **PanEcho**: Pan-view echocardiography analysis model

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



## Model Information

### EchoPrime
- **Repository**: [https://github.com/ouwen/EchoPrime](https://github.com/ouwen/EchoPrime)
- **Hugging Face**: [https://huggingface.co/echoprime/echoprime-base](https://huggingface.co/echoprime/echoprime-base)


### R3D-Transformer
- **Repository**: [https://github.com/kenshohara/3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch)


### PanEcho
- **Repository**: [https://github.com/echocardiography/PanEcho](https://github.com/echocardiography/PanEcho)




## Related Publications

- Johnson, A. E. W., et al. "MIMIC-IV, a freely accessible electronic health record dataset." Scientific Data 10.1 (2023): 1.
- Goldberger, A. L., et al. "PhysioBank, PhysioToolkit, and PhysioNet: components of a new research resource for complex physiologic signals." Circulation 101.23 (2000): e215-e220.



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support and Contact

- **Documentation**: [Full documentation](https://github.com/MIT-LCP/vector-embedding/wiki)
- **Issues**: [GitHub Issues](https://github.com/MIT-LCP/vector-embedding/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MIT-LCP/vector-embedding/discussions)
- **Contact**: [MIT-LCP Team](mailto:mimic-support@physionet.org)

## Partner Organizations

| [<img src="https://physionet.org/static/images/physionet-logo.png" alt="PhysioNet" width="200"/>](https://physionet.org/) | [<img src="https://upload.wikimedia.org/wikipedia/commons/0/0c/MIT_logo.svg" alt="MIT" width="150"/>](https://www.mit.edu/) | [<img src="https://www.khidi.or.kr/cms/image/layout/lnb_logo.png" alt="KHIDI" width="200"/>](https://www.khidi.or.kr/) |
|:---:|:---:|:---:|
| PhysioNet | MIT | KHIDI |

</div>

## Acknowledgments
- **Funding**: This research was supported by a grant of the Korea Health Technology R&D Project through the Korea Health Industry Development Institute (KHIDI), funded by the Ministry of Health & Welfare, Republic of Korea (grant number: RS-2024-00439677)

---

**Disclaimer**: This software is provided for research purposes only. It is not intended for clinical use. Always comply with your institution's ethics and data usage policies when working with healthcare data.