# SceneTextDetection

1) The data we used on consists of around 200 images half - dark text with bright background and half the opposite in the link 
below - https://drive.google.com/open?id=1nUXD9dgbnWq071y-uXI35eXGLlkywYzR

2) The code is run by the code.py file then we input the image location and then if text is dark on bright background or the opposite when prompted

3) We will get the output images with letter wise, line wise and word wise bounding boxes for the image.

4) The outputs for all the above images is in this link - https://drive.google.com/open?id=1xw3l873lwPtdgJOQ4vCBveazlMk3pSca

5) There is a CNN model trained on ICDAR 2003 dataset for removing false positives in our letter wise segmentation whose saved model is provided.

6) To relearn the model run the character_recognition.py file you will be prompted for location of the folder with the icdar 2003 images which you can get from here - https://drive.google.com/open?id=1XHGR-5hgaeK1op-0yXv8QUW9WXEo_sOM and the label xml file is already in our repository
