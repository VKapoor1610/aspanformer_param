# ASpanFormer + Classifier Model Implementation 

## Installation 
```bash
conda env create -f environment.yaml
conda activate ASpanFormer
```

## Get started
Download aspanformer weights from [AspanFormer Weights](https://drive.google.com/file/d/1eavM9dTkw9nbc-JqlVVfGPU5UvTTfc6k/view?usp=share_link)  
Download classifier weights from [Classifier Weights](https://drive.google.com/file/d/1LKKF_gYVbZUvyKuK_Mvw8WbV0nqMMTbL/view?usp=sharing)

Extract weights by
```bash
tar -xvf weights_aspanformer.tar
```
Classifier weights are already extracted.

A demo to match one image pair is provided. To get a quick start, 

```bash
cd demo
python demo.py
```

## Training the model 
To train the model go to forehead-recognition.ipynb file and run all the cells, this will ensure all dependencies are installed and will the train the model.
The training has been done on kaggle notebook and requires certain datasets to be uploaded, since the datasets are not availalble in public please upload the FHV1 dataset available with Dr. Aditya to replicate the results.
the training weights for the classifer model can be accessed from [Classifier Weights](https://drive.google.com/file/d/1LKKF_gYVbZUvyKuK_Mvw8WbV0nqMMTbL/view?usp=sharing)


## Sample run 
Our downstream task is to predict whether two images are of same person or not! 
To find the same simply upload two images from dataset and run scorefile.py outputOfModel function which takes two image paths as input and returns the ouput probability whether two images are same or not.

```python 
import scorefile.outputOfModel as outputOfModel

result , prob = outputOfModel( Aspanformer , Classifier , img0_path , img1_path )
# generates the result and prob. 
```

