# Building Extraction From Aerial Imagery

This project demonstrates Aerial Imagery Building Map Feature Extraction. 
Training data-set was used from [INRIA](https://project.inria.fr/aerialimagelabeling/) 
and training was performed using [PyTorch](https://pytorch.org/)

## Training Details

- Model Used for Training : [RefineNet Lite](https://arxiv.org/pdf/1810.03272.pdf)
- Optimizer : Adam
    - weight decay of 0.0005 was chosen, after validation
- Augmentation Applied : Color and Geometric Transformation
- Loss Function :  Binary Cross Entropy (BCE) + Jaccard 
    - Weight distribution is as follows, *0.8* for BCE and *0.2* for Jaccard

## Inference Details

- Test Time Augmentation _(TTA)_ were performed on the test image during inference.
Library used for [TTA](https://github.com/cypherics/ttAugment)



👈 **Please select _DEMO_ in the sidebar to start.**

