# How To Create and Setup A Data Science VM in Azure
This is a step by step tutorial on how to setup an Azure Data Science VM (Ubuntu 18.04)
to use for running Python scripts and Jupyter Lab notebooks.

## Login to Azure Portal
You can login to Azure Portal at [https://portal.azure.com](https://portal.azure.com)

## Create A Resource Group
Resource groups allow you to organize related resources so that you can easily find and manage them.
For example, a VM and a data disk for that VM would be placed into the same resource group.
If you have already created a resource group that you would like to use, you can skip this section.

1. Search for 'Resource groups' and click on 'Resource groups'
2. Click '+ Add'
3. TODO


## Create Azure Data Science VM

1. Search for 'Data Science Virtual Machine' and choose 'Data Science Virtual Machine - Ubuntu 18.04'
2. Click 'Create'
3. TODO

## SSH Into VM

1. Copy public IP address of the VM under the 'Overview' section on the VM page.
2. In a terminal enter `ssh <username>@<public IP>`
3. Enter your password for the VM.
    * You can change your username and password for the VM on the VM page under 'Reset password'
4. TODO

## Configure Jupyter
In the terminal where you have made your remote SSH connection:

1. `sudo nano /etc/jupyterhub/jupyterhub_config.py`
    * You may be asked to enter your password
2. Add the line `c.Spawner.default_url = '/lab'`
    * This will default Jupyter to start in lab instead of hub
3. TODO: Configure Jupyter to start in ~

## Create Anaconda Environment

## Install Anaconda Environment As Jupyter Kernel

1. Activate conda environment `conda activate <env name>`
2. `ipython kernel install --user --name=<env name>`

## Configure Data Disk

### Identify the data disk

1. `lsblk`

### Format the data disk

2. `sudo fdisk /dev/<disk>`
3. `g`
4. `n`
5. <Enter>
6. <Enter>
7. `w`

### Create the file system

1. `sudo mkfs -t ext4 /dev/<partition name>`

### Mount the data disk

1. `sudo mkdir /data`
2. `sudo mount /dev/<partition name> /data`
3. TODO: permissions?

### Setup data disk to mount on boot

1. `sudo nano /etc/fstab`
2. TODO:
