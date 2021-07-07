# UrbEX

### Tensorflow env

      conda create --name tf1 python=3.7
      conda activate tf1
      conda install -c conda-forge tensorflow-gpu=1.14 cudatoolkit=10.0 cudnn=7.3.1
      conda install -c conda-forge tensorflow-gpu=1.15
      
      conda create --name tf2 python=3.8
      conda install -c conda-forge tensorflow-gpu=2.3 cudatoolkit=11.0
      
      conda install numpy matplotlib scikit-learn pandas tqdm scikit-image
      conda install -c conda-forge tensorboardx
      conda install -c conda-forge opencv
      conda install -c conda-forge scipy=1.5.3
      conda install -c conda-forge notebook
      conda deactivate

