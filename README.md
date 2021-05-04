# CXR-Net

CXR-Net: An Artificial Intelligence Pipeline for Quick Covid-19 Screening of Chest X-Rays 

Haikal Abdulah 1,4, Benjamin Huber 1,4, Sinan Lal 1, Hassan Abdallah 3, Luigi L. Palese 4, Hamid Soltanian-Zadeh 5, Domenico L. Gatti 1,2,6*

1 Department of Biochemistry, Microbiology and Immunology, Wayne State Univ., Detroit, MI, USA 

2 NanoBioScience Institute, Wayne State Univ., Detroit, MI, USA

3 Department of Biostatistics, University of Michigan, Ann Arbor, MI, USA  

4 Department of Basic Medical Sciences, Neurosciences and Sense Organs, Univ. of Bari Aldo Moro, Bari, Italy

5 Departments of Radiology and Research Administration, Henry Ford Health System, Detroit, MI, USA

6 Molecular Medicine Institute, Cambridge, MA 02138,USA


*E-mail: dgatti@med.wayne.edu


CXR-Net is a two-module Artificial Intelligence pipeline for the quick detection of SARS-CoV-2 from Antero/Posterior (A/P) chest X-rays (CXRs). Module 1 was trained on a public dataset of 6395 A/P CXRs with radiologist annotated lung contours to generate masks of the lungs that overlap the heart and large vasa. Module II is a hybrid convnet in which the first convolutional layer with learned coefficients is replaced by a layer with fixed coefficients provided by the Wavelet Scattering Transform (WST). Module 2 takes as inputs the patients’ CXRs and corresponding lung masks calculated by Module I, and produces as outputs a class assignment (Covid vs. non-Covid) and high resolution heat maps that identify the SARS associated lung regions. Module 2 was trained on a dataset of CXRs from non-Covid and RT-PCR confirmed Covid patients acquired at the Henry Ford Health System (HFHS) Hospital in Detroit. All non-Covid CXRs were from pre-Covid era (2018-2019), and included images from both normal lungs and lungs affected by non-Covid pathologies. Training and test sets consisted of 2265 CXRs (1417 Covid –, 848 Covid +), and 1532 CXRs (945 Covid –, 587 Covid +), respectively. Six distinct cross-validation models with the same Covid +/– ratio, each trained on 1887 images and validated against 378 images, were combined into an ensemble model that was used to classify the CXR images of the test set with resulting Accuracy = 0.789, Precision = 0.739, Recall = 0.693, F1 score = 0.715, ROC(auc) = 0.852.

The repository contains two folders, Module_1 and Module_2, with all the jupyter notebooks that were used for training and validation. There are only example images for inputs and outputs. The folder run_CXR-Net contains two jupyter notebooks, Predict_new_patient_from_processed_png.ipynb and Predict_new_patient_from_unprocessed_dcm.ipynb, plus additional folders that can be used to run the entire CXR-Net pipeline with the provided example image. Please notice that with the release of Kymatio (0.3.dev0) used to prepare this version of CXR-Net, the ensemble model must be generated by first compiling separately the Wavelet Scattering Transform model and then joining it to the ensemble resnet model. The  next release of Kymatio will allow the entire ensemble model to be saved on file and reloaded in memory when needed. Currently, the ensemble model compilation takes about 2 minutes, but once the model is loaded in memory, the class prediction (Covid positive or negative, with corresponding scores) and the heat map generation take about 2 seconds per image with 2 V100 GPUs.

Please, refer to the manuscript in ArXiv for additional information on how the JMS and V7 databases of CXRs with segmented lungs were generated. These databases are available on request from the corresponding author.

The database of Covid negative and Covid positive CXRs used in this study is an exclusive property of the Henry Ford Health System in Detroit, and is not publicly available.


For additional information e-mail to:

Domenico Gatti

Dept. Biochemistry, Microbiology, and Immunology, Wayne State University, Detroit, MI.

E-mail: dgatti@med.wayne.edu

website: http://veloce.med.wayne.edu/~gatti/
