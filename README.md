# CXR-Net

A Hybrid Pipeline for Covid-19 Screening Incorporating Lungs Segmentation and Wavelet Based Preprocessing of Chest X-Rays

Haikal Abdulah 1,4, Benjamin Huber 1,4, Sinan Lal 1, Hassan Abdallah 3, Luigi L. Palese 4, Hamid Soltanian-Zadeh 5, Domenico L. Gatti 1,2,6*

1 Department of Biochemistry, Microbiology and Immunology, Wayne State Univ., Detroit, MI, USA 

2 NanoBioScience Institute, Wayne State Univ., Detroit, MI, USA

3 Department of Biostatistics, University of Michigan, Ann Arbor, MI, USA  

4 Department of Basic Medical Sciences, Neurosciences and Sense Organs, Univ. of Bari Aldo Moro, Bari, Italy

5 Departments of Radiology and Research Administration, Henry Ford Health System, Detroit, MI, USA

6 Molecular Medicine Institute, Cambridge, MA 02138,USA


*E-mail: dgatti@med.wayne.edu


CXR-Net is a two-module pipeline for the detection of SARS-CoV-2 from chest X-rays (CXRs). Module 1 is a traditional convnet that generates masks of the lungs overlapping the heart and large vasa. Module 2 is a hybrid convnet that preprocesses CXRs and corresponding lung masks by means of the Wavelet Scattering Transform, and passes the resulting feature maps through an Attention block and a cascade of Separable Atrous Multiscale Convolutional Residual blocks to produce a class assignment as Covid or non-Covid. Module 1 was trained on a public dataset of 6395 CXRs with radiologist annotated lung contours. Module 2 was trained on a dataset of 2362 non-Covid and 1435 Covid CXRs acquired at the Henry Ford Health System Hospital in Detroit. Six distinct cross-validation models, were combined into an ensemble model that was used to classify the CXR images of the test set. An intuitive graphic interphase allows for rapid Covid vs. non-Covid classification of CXRs, and generates high resolution heat maps that identify the affected lung regions.

https://medrxiv.org/cgi/content/short/2022.03.13.22272311v1

<img width="248" alt="image" src="https://user-images.githubusercontent.com/32550835/158189043-39d62428-0326-4153-8fef-5016988b0e01.png">

The repository contains two folders, Module_1 and Module_2, with all the jupyter notebooks that were used for training and validation. There are only example images for inputs and outputs. The folder run_CXR-Net contains two jupyter notebooks, Predict_new_patient_from_processed_png.ipynb and Predict_new_patient_from_unprocessed_dcm.ipynb, plus additional folders that can be used to run the entire CXR-Net pipeline with the provided example image. Please notice that with the release of Kymatio (0.3.dev0) used to prepare this version of CXR-Net, the ensemble model must be generated by first compiling separately the Wavelet Scattering Transform model and then joining it to the ensemble resnet model. The  next release of Kymatio will allow the entire ensemble model to be saved on file and reloaded in memory when needed. Currently, the ensemble model compilation takes about 2 minutes, but once the model is loaded in memory, the class prediction (Covid positive or negative, with corresponding scores) and the heat map generation take about 2 seconds per image with 2 V100 GPUs.

A Python script, Predict_new_patient.py, can be used to select from the command line both source/target directories, and, optionally, which dcm image files must be processed for covid/non-covid classification. 

A GUI application, g-CXR-Net is available in a separate repository of dgattiwsu. A video describing the use of g-CXR-Net can be watched/downloaded at http://veloce.med.wayne.edu/~gatti/neural-networks/cxr-net.html.

A brief medRxiv article describing the features of g-CXR-Net is available at https://medrxiv.org/cgi/content/short/2021.06.06.21258428v1, or by mobile device's via the QR code shown below:

<img width="248" alt="image" src="https://user-images.githubusercontent.com/32550835/123811248-1191fd00-d8c1-11eb-8302-9514a6d7197b.png">


Please, refer to the manuscript in medrxiv (https://medrxiv.org/cgi/content/short/2022.03.13.22272311v1) for additional information on how the JMS and V7 databases of CXRs with segmented lungs were generated. These databases are available on request from the corresponding author.

The database of Covid negative and Covid positive CXRs used in this study is an exclusive property of the Henry Ford Health System in Detroit, and is not publicly available.


For additional information e-mail to:

Domenico Gatti

Dept. Biochemistry, Microbiology, and Immunology, Wayne State University, Detroit, MI.

E-mail: dgatti@med.wayne.edu

website: http://veloce.med.wayne.edu/~gatti/
