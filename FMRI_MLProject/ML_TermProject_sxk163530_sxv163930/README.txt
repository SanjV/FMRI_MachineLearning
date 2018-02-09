
To run any algorithm: just call python _nameOfFile.
extract the data and place them in the folder where the code is placed.
To change what data the algorithm is trained and tested on, you need to go into the file and change
the parameters in the call to util.getVoxelArray(). The parameters to that function control are outlined in the function signature and they control which subjects and which tasks you want to have in the dataset. 
Attached is a matlab file containing 3 matrices for your project:
OB, WM, SA

OB=oddball task
WM=working memory task
SA=selective attention task

Each matrix is 3 dimensional voxels x conditions x runs containing the amplitude of response of each voxel to a certain condition and run
voxels: There are 1973 voxels taken from one subjects ventral temporal cortex (VTC); The voxels are in the same order in all matrices.
conditions: There are 5 conditions in the following order: Faces, bodies, cars, houses, and words.
runs: There are 3 runs. Each subject participated in 3 runs of each task using different images from the same category.

Example: 
OB(:,1,1) reflects the distributed responses of VTC voxels to faces in the first run of the oddball task
WM (:,3,2) reflects the distributed responses of VTC to cars in the second run of the working memory task



