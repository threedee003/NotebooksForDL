## Unsupervised Image Segmentation for N channelled images.

*This project was made by Tribikram Dhar, Dept of Electrical Engg, Jadavpur University.*


* The algorithm takes an input (N x C x H) and generates an single chanelled segmented mask of the image in O(NHC) time.
* The distance function is customizable, so are the number of epochs, error threshold.

#### Original Image

![A sample image of an airport](./images/airport.jpeg)


| Number of epochs |                                 Segmented Image                                                |
|  --------------  |  --------------------------------------------------------------------------------------------  |
| Untrained (Epochs = 0) |   ![](./images/1_ut.png)                                                                       |
| Epochs = 10      |   ![](./images/1_t_10eps.png)                                                                  |
| Epochs = 15      |   ![](./images/1_t_15eps.png)                                                                  |
| Epochs = 25      |   ![](./images/1_t_25eps.png)                                                                  |


*the documentation is under development*


  
