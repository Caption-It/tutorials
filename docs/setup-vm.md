# How To Create and Setup A Data Science VM in Azure
This is a step by step tutorial on how to setup an Azure Data Science VM (Ubuntu 18.04)
to use for running Python scripts and Jupyter Lab notebooks.
Throughout this guide we use the convention that `<>` indicates a value that must be replaced
with a value for your system. e.g. You should replace `<username>`, with your username.

## Create an Azure For Students Account
1. Go to [https://azure.microsoft.com/en-us/free/students](https://azure.microsoft.com/en-us/free/students).
2. Click 'Activate now'.
3. Follow the steps to enter your phone number, student email, sign into your microsoft account, and verify your student account.
    * If you don't already have a Microsoft account, you will need to create one.

## Login to Azure Portal
You can login to Azure Portal at [https://portal.azure.com](https://portal.azure.com).
Be sure to use the Microsoft account that you associated with your Azure for Students account.

## Create A Resource Group
Resource groups allow you to organize related resources so that you can easily find and manage them.
For example, a VM and a data disk for that VM would be placed into the same resource group.
If you have already created a resource group that you would like to use, you can skip this section.

1. Search for 'Resource groups' and click on 'Resource groups'.
2. Click '+ Add'.
3. For 'Subscription' choose 'Azure for Students'.
4. For 'Resource group' enter any name you choose. e.g. demo-rg.
5. For 'Region' choose '(US) West US 2'.
6. Click 'Review + create'.
7. Click 'Create'.


## Create Azure Data Science VM

1. Search for 'Data Science Virtual Machine' and choose 'Data Science Virtual Machine - Ubuntu 18.04'.
2. Click 'Create'.
3. For 'Subscription' choose 'Azure for Students'.
4. For 'Resource group' choose the resource group you already created or enter the name of a new one.
5. For 'Virtual machine name' enter any name you want that meets the criteria. e.g. student-demo-vm-1.
6. For 'Region' choose '(US) West US 2'.
7. For 'Size' click 'Select size'.
8. Click 'clear all filters'.
9. Search for 'NC' and choose one of the NC*_Promo VMs. I recommend starting with NC6_Promo since it is the cheapest.
10. Click 'Select'.
11. For 'Authentication type' choose 'Password'.
12. For 'Username' choose a short memorable name. e.g. casey.
13. For 'Password' choose a password that meets the criteria.
14. For 'Confirm Password' re-enter the same password as in the previous step.
15. Click 'Next : Disks'.
16. For 'OS disk type' choose 'Standard SSD'.
17. Click 'Create and attach new disk'.
18. For 'Size' click 'Change size'.
19. Choose '512 GiB'.
20. Click 'OK'.
21. Click 'OK'.
22. Click 'Next : Networking'.
23. Click 'Next : Management'.
24. Click 'Next : Advanced'.
25. For 'Extensions' click 'Select an extension to install'.
26. Select 'NVIDIA GPU Driver Extension'.
27. Click 'Create'.
28. Click 'OK'.
29. Click 'Review + create'.
30. Click 'Create'.

Make sure you stop your VM after you are done using it or else you will continue to get charged for it.

## SSH Into VM

1. Copy public IP address of the VM under the 'Overview' section on the VM page.
2. In a terminal enter `ssh <username>@<public IP>`
3. Enter your password for the VM.
    * You can change your username and password for the VM on the VM page under 'Reset password'.

You now are in a remote terminal to the VM. Entering commands here will get executed on the VM.

## Configure Jupyter
In the terminal where you have made your remote SSH connection:

1. `sudo nano /etc/jupyterhub/jupyterhub_config.py`
    * You may be asked to enter your password
2. Add the line `c.Spawner.default_url = '/lab'`
    * This will default Jupyter to start in lab instead of hub
3. Find the line that has `c.Spawner.notebook_dir = '~/notebooks'` and change it to `c.Spawner.notebook_dir = '~/'`.
4. `ctrl+o`
5. `Enter`
6. `ctrl+x`

You may need to restart the VM and establish a new ssh terminal after these steps.

## Clone The Code
In a remote ssh terminal to your VM:

1. `git clone <clone url> <new folder>`

## Create Anaconda Environment
In a remote ssh terminal to your VM:

1. `cd <repo folder>`
1. `conda env create -f environment.yml`

## Install Anaconda Environment As Jupyter Kernel
In a remote ssh terminal to your VM:

1. Activate conda environment `conda activate cap-env`
2. `ipython kernel install --user --name=cap-env`

## Configure Data Disk

### Identify the data disk
In a remote ssh terminal to your VM:

1. `lsblk`

Look in the output for a disk that does not have a mounted partition.
That is the name of the data disk you will configure.
Usually this will be 'sdc' if you follow this guide.

### Format the data disk

2. `sudo fdisk /dev/<disk>`
3. `g`
4. `n`
5. `Enter`
6. `Enter`
7. `w`

### Create the file system

1. `sudo mkfs -t ext4 /dev/<partition name>`

The parition name is usually the name of the disk + a number.
you can use `lsblk` to find the name of the partition.
If you are following this guide, it will usually be 'sdc1'

### Mount the data disk

1. `sudo mkdir /data`
2. `sudo chown <username> /data`
3. `sudo mount /dev/<partition name> /data`

### Setup data disk to mount on boot

1. `sudo echo "/dev/<partition name>        /data   auto    defaults,nofail 0 0" >> /etc/fstab`

You may need to reboot the VM for this to take effect.

## Go To Jupyter Lab

In a browser go to, `https://<public ip>:8000`.
You will be prompted for a username and password. Enter the same username and password you use for you VM.
