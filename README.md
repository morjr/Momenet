# Momenet

We propose a new network that favors geometric moments for point cloud object classification. 
The most prominent element of the network is supplementing the given point cloud coordinates  together with polynomial functions of the coordinates.
This simple operation allows the network to account for higher order moments of a given shape.   
![](https://github.com/morjr/Momenet/blob/master/figures/arch.PNG?raw=true)

# Citation
Please cite our paper:  

	@inproceedings{joseph2019momen,
    title={Momen(e)t: Flavor the moments in learning to classify shapes},
    author={Joseph-Rivlin, Mor and Zvirin, Alon and Kimmel, Ron},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    pages={0--0},
    year={2019}
     }

# Running Instructions

### Classic Momenet

To train a model run: python train.py
To test the model run: python evaluate.py

#### Dataset - <a href="http://modelnet.cs.princeton.edu/" target="_blank">ModelNet40</a>
For linux users: the dataset will be automatically downloaded (416MB) to the data folder.  
For window users: download the zip from https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip and extract it under `data` folder.

### Parameters
**model** is the filename in the models folder. The default is momenet.  
**log** is the output folder.  
**num_point** is the number of points per point cloud. The default is 1024. 

### Momenet with nearest neighbors:
To train a model run: python momenet_NN\train.py


# Acknowledgement
The structure of this code is from PointNet.
