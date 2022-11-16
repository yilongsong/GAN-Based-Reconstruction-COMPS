# GAN-Based-Reconstruction-COMPS

Replication of "Image-Adaptive Generative Adversarial Netowrks" (IAGANs), proposed in (Hussein et al., 2020), in PyTorch. 

Antonio, Yilong, and Daisuke (Fall 2022)

## Description

Our work examines the image-adaptive generative model, proposed in (Hussein et al., 2020), that purports to mitigate problems caused by the limited representation capabilities of previous models in solving the image inverse problem. To this end, we implement the proposed model from (Hussein et al., 2020) and evaluate the robustness of their methodologies. Our results show that the two approaches, IA and BP, can effectively improve reconstructions. Particularly, the image adaptation technique (i.e., IA) is very effective in enhancing the generator’s capability to estimate the specific test sample. Further investigatin reveals slight biases in the model, which we conjecture to be caused by the training dataset for the PGGAN and thus able to be ameliorated by updating the dataset. Finally, to explore more efficient ways of running the model, we tested out our conjecture that the number of iterations used for CSGM can be lowered, and find that it indeed can be lowered to an extent without sacrificing model performance. However, future work can further explore the optimal number of iterations for IA. We conjecture that performing too many IA iterations will produce worse reconstructions, due to “overfitting” of the model, and hence hinder its ability to generate a realistic image of human faces.

## Usage
To run our model , you can use the command line arguments as follows:

``` !python main.py --GAN PGGAN --scale 32 --noise 40 --task Bicubic --csgm_itr 1800 --ia_itr 300 --test_folder Whole  --save_images ```

## References
Karras, Tero, Timo Aila, Samuli Laine, and Jaakko Lehtinen. "Progressive growing of gans for improved quality, stability, and variation." arXiv preprint arXiv:1710.10196 (2017).

Hussein, Shady Abu, Tom Tirer, and Raja Giryes. "Image-adaptive GAN based reconstruction." In Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34, no. 04, pp. 3121-3129. 2020.

Bora, Ashish, Ajil Jalal, Eric Price, and Alexandros G. Dimakis. "Compressed sensing using generative models." In International Conference on Machine Learning, pp. 537-546. PMLR, 2017.

Zhang, Richard, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. "The unreasonable effectiveness of deep features as a perceptual metric." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 586-595. 2018.

Goodfellow, Ian, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. "Generative adversarial networks." Communications of the ACM 63, no. 11 (2020): 139-144.
