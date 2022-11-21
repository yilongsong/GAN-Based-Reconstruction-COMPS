# GAN-Based-Reconstruction-COMPS

Replication of "Image-Adaptive Generative Adversarial Netowrks" (IAGANs), proposed in (Hussein et al., 2020), in PyTorch. 

Antonio, Yilong, and Daisuke (Fall 2022)

## Description

The image inverse problem is the problem of reconstructing an image given its degraded 
or compressed observation. Some previous solutions to this problem use generative adversarial 
networks (GANs), but the representation capabilities of such models cannot capture the full 
distribution of complex classes of images (e.g., human faces), thus producing sub-optimal results. 
Our work examines the image-adaptive generative model, proposed in Hussein et al (2020), that 
purports to mitigate the limited representation capabilities of previous models in solving the 
image inverse problem. To this end, we implement the proposed model from Hussein et al (2020), 
which makes generators "image-adpative" to a specific test sample. This model consists of three 
successive optimization stages: the non-image-adaptive "compressed sensing using generative models" 
(CSGM), the image-adaptive step (IA), and the post-processing "back-projection" (BP). Our results 
demonstrate that the two image-adaptive approaches--IA and BP--can effectively improve reconstructions. 
Further testing reveals slight biases existing in the model (e.g., skin tones), which we conjecture to 
be caused by the training dataset on which the model is trained. Finally, to explore more efficient ways 
of running the model, we test out different numbers of iterations used for CSGM. The results show that we 
can indeed decrease the number of CSGM iterations without compromising reconstruction qualities. 

## Website and Paper
Coming soon.

## Usage
To run our model , you can use the command line arguments as follows:

``` 
!python main.py --GAN PGGAN --scale S --noise N --task T --csgm_itr CSGM --ia_itr IA --test_folder FOLDER  --save_images 
```

where you replace S with scale, N with noise level, T with a task, CSGM with the number of iterations for CSGM, IA with the number of iterations for IA, and FOLDER with the set of images you want to run your experiment on. You can also use the Jupyter notebook to run our model.

## References
Karras, Tero, Timo Aila, Samuli Laine, and Jaakko Lehtinen. "Progressive growing of gans for improved quality, stability, and variation." arXiv preprint arXiv:1710.10196 (2017).

Hussein, Shady Abu, Tom Tirer, and Raja Giryes. "Image-adaptive GAN based reconstruction." In Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34, no. 04, pp. 3121-3129. 2020.

Bora, Ashish, Ajil Jalal, Eric Price, and Alexandros G. Dimakis. "Compressed sensing using generative models." In International Conference on Machine Learning, pp. 537-546. PMLR, 2017.

Zhang, Richard, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. "The unreasonable effectiveness of deep features as a perceptual metric." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 586-595. 2018.

Goodfellow, Ian, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. "Generative adversarial networks." Communications of the ACM 63, no. 11 (2020): 139-144.
