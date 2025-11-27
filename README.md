# B4DL \[[Paper](https://www.arxiv.org/pdf/2508.05269)\]
Official PyTorch implementation of the paper "B4DL: A Benchmark for 4D LiDAR LLM in Spatio-Temporal
Understanding".

---

## Installation

We recommend setting up a conda environment for the project:
```shell
conda create --name=vtimellm python=3.10
conda activate vtimellm

git clone https://github.com/huangb23/VTimeLLM.git
cd VTimeLLM
pip install -r requirements.txt
```
Additionally, install additional packages for training cases.
```shell
pip install ninja
pip install flash-attn --no-build-isolation
```
## Dataset

## Training

For training instructions, check out [train.md](docs/train.md).



## Acknowledgements
This work was partly supported by the Institute of Information &
Communications Technology Planning & Evaluation(IITP) grant
funded by the Korea government(MSIT) (No.RS-2024-00439020,
Developing Sustainable, Real-Time Generative AI for Multimodal
Interaction, SW Starlab) and partly supported by the Institute of
Information & Communications Technology Planning & Evaluation(IITP) grant funded by the Korea government(MSIT) (No.RS2025-02283048, Developing the Next-Generation General AI with
Reliability, Ethics, and Adaptability)

If you're using VTimeLLM in your research or applications, please cite using this BibTeX:
```bibtex
@inproceedings{Choi_2025, series={MM ’25},
   title={B4DL: A Benchmark for 4D LiDAR LLM in Spatio-Temporal Understanding},
   url={http://dx.doi.org/10.1145/3746027.3755074},
   DOI={10.1145/3746027.3755074},
   booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
   publisher={ACM},
   author={Choi, Changho and Shin, Youngwoo and Han, Gyojin and Lee, Dong-Jae and Kim, Junmo},
   year={2025},
   month=oct, pages={3399–3407},
   collection={MM ’25} }
```

## License :scroll:
<a rel="license" href="https://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/80x15.png" /></a> 

This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License</a>.
