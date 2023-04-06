# text-to-image-generation

This is the repository of Pseudo Lab's Text-to-Image Generation (feat. Diffusion) team.

:bulb: Our aim is to review papers and code related to image generation and text-to-image generation models, approach them theoretically, and conduct various experiments by fine-tuning Diffusion based models.

[About Us - Pseudo Lab](https://www.linkedin.com/company/pseudolab/)

[About Us - Text-to-Image Generation (feat. Diffusion) Team](https://pseudo-lab.com/Text-to-Image-Generation-feat-Diffusion-cc12047d1bfc4bdfa70122c11ff90aee)

참여 방법: 매주 수요일 오후 9시, 가짜연구소 Discord Room-DH 로 입장!

## Contributor 
- 조상우 [Sangwoo Jo] | [Github](https://github.com/jasonjo97) | [Linkedin](https://www.linkedin.com/in/sangwoojo/) | 
- 문광수 [] | |
- 김지수 [] | |

## Schedule 
| idx | Date | Presenter | Paper / Code | 
| :--: | :--: | :--: | :--: |
| 1 | 2023.03.29 | 조상우 [Sangwoo Jo] | [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) (ICLR 2014) <br> [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) (NIPS 2014)| 
| 2 | 2023.04.05 | 문광수 [] <br> 김지수 [] | [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) (ICCV 2017) <br> [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) (CVPR 2019)| 

## Jupyter Book Update Procedure  
1. Clone the repo on your local computer  
```bash
git clone https://github.com/Pseudo-Lab/text-to-image-generation.git
```

2. Install required packages 
```bash
pip install jupyter-book==0.15.1
pip install ghp-import==2.1.0
```

3. Change the contents in ```book/docs``` folder and ```_toc.yml``` file accordingly 
```
format: jb-book
root: intro
parts:
- caption: Paper/Code Review
  chapters:
  - file: docs/review/vae
  - file: docs/review/gan
```

4. Build the book using Jupyter Book command
```bash
jupyter-book build ./book
```

5. Sync your local and remote repositories
```bash
cd text-to-image-generation
git add .
git commit -m "adding my first book!"
git push
```

6. Publish your Jupyter Book with Github Pages
```
ghp-import -n -p -f book/_build/html -m "initial publishing"
```
