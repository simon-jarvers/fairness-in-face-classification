# Exploring Fairness in the Classification of Faces

Deep Learning for Data Science (DD2424) KTH spring 2023

Project: Exploring Fairness in the Classification of Faces

Contributors: Annika Oehri, Frawa Vetterli, Simon Jarvers

Code remarks: dlds_code.py contains our main training code. To do the fairness analysis and plots, refer to prediction_analysis.py

## Abstract

This project addresses the issue of fairness in machine learning, particularly in the
context of face classification tasks. We use the FairFace dataset, which aims to
alleviate biases and promote balanced representation across different races. Deep
residual networks (ResNets) are trained on this dataset, utilizing both transfer learning
and hyperparameter optimization for the model’s fine-tuning. Additionally, we
employ data augmentation techniques such as classic augmentation, Mix Up, and
Cut Mix to improve accuracy and mitigate overfitting. To jointly address gender
and race classification, we adopt a multi-task learning framework, enabling knowledge
sharing between related tasks. The results are analyzed in relation to fairness,
and various strategies are implemented at both the data and loss levels to address
issues of class imbalance unfairness and disparities in class prediction accuracies.
Our experiments assess the effectiveness of our proposed methods in achieving
the desired outcomes for this specific task. Encouragingly, the experimental results
demonstrate that we were able to enhance fairness without experiencing any
substantial performance decline.


## FairFace Dataset

The FairFace dataset (https://github.com/joojs/fairface) addresses bias in face attribute recognition tasks through balanced gender
and race representation, improved generalization to unseen data, inclusion of underrepresented groups
and thus helps mitigating biases in commercial systems. Overall, fairness and the FairFace dataset
contribute to building less biased and fairer neural networks in face recognition tasks, promoting
ethical and reliable automated systems.
The dataset consists of unprocessed images featuring faces in natural settings, lacking any alignment
and displaying various angles, potential occlusions, and other factors. The provided labels include
age, gender, and race, from which we worked with gender and race. The full unbalanced version
of the dataset has a total of 97696 images, with overproportional representation of white men and
women and underproportional representation of middle eastern women. One can also filter the dataset
to a balanced version with same number of images per gender and race class with a total of 45414
images. We split the data into 78% training, 11% validation and 11% test sets.
Personal remarks: It’s important to acknowledge that classifying gender and race solely based on facial appearance
is inherently problematic and overly simplistic. For gender, we had to adopt the dataset’s binary
approach, although it fails to encompass the entire spectrum of gender identities and is confounded
with biological sex, which also extends beyond the traditional male and female categories as suggested
by the data. Similarly, the dataset’s categorization into seven races is overly simplified. Moreover, the
term "race" lacks biological significance when applied to humans, and in the context of this dataset,
"ethnicity" might be a more suitable term to use.


### Samples of the FairFace Datatset
<figure>
  <img
  src="./readme_resources/face_samples.JPG?raw=true"
  alt='Face Images taken "in the wild"'>
</figure>

## Network

We used ResNets provided by the torchvision library of Pytorch pretrained on the ImageNet dataset.
Used ML techniques: Transfer Learning, Multi-Task Learning (for Race and Gender), Data Augmentation (Classical: flipping, rotation, translation, random color jitter to adjust brightness, contrast, saturation; Mix Up; Cut Mix)


## Fairness Evaluation and Improvement

1. Although unbalanced datasets often yield biased networks the inversion of the argument cannot be assumed. A balanced dataset is not a suffcient criterion for an unbiased model.
2. We used different approaches to improve fairness between classes (e. g. minimizing the standard deviation of per class accuracies). The most promising method was an attention based technique rating underperforming classes higher and overperforming classes lower.

### Calssification performance without attention scores
<figure>
  <img src="./readme_resources/plot_normalize_True_predictions_config_combined_bal_aug_2023-05-17_17-02-07_416000.png?raw=true" alt="Without attention score"/> 
</figure>

### Calssification performance with attention scores
<figure>
  <img src="./readme_resources/plot_normalize_True_predictions_config_combined_bal_aug_valattention_2023-05-18_13-06-33_814758.png?raw=true" alt="With attention score"/>
</figure>


## Conclusion

We have shown that standard methods in image classification in form of ResNets combined with
data augmentation and multi-task learning can also be applied to our specific task of gender and race
classification of the FairFace dataset. The results demonstrate that the mere utilization of balanced
data sets does not guarantee the development of fair models, as certain classes may inherently pose
greater challenges for learning. Conversely, the implementation of a weighted loss, based on the
prediction imbalances observed in the validation set, is shown to be more effective in generating fairer
outcomes in the sense of reducing inter-class variance. Alternative approaches employing loss-based
methods to achieve fairness did not yield successful results. Nevertheless, these findings suggest that
optimization-targeted methods, particularly loss modifications, have the potential to rectify unfair
models, although potentially at the expense of overall accuracy. However, in this particular instance,
we were able to keep our level of accuracy while improving fairness of the model’s predictions.
