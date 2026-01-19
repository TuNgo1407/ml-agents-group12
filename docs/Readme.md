#  Group 12 | Project 2.1 — AI and Machine Learning in the Unity Game Engine 

## Overview

This repository contains the work for our Project 2.1.  
Our objective is to collect data from reinforcement learning training runs conducted in Unity ML-Agents and use this data to train machine learning models capable of predicting various properties of deep RL training sessions.

In **Phase 1**, our focus is on:
- Setting up a project environment (Unity + ML-Agents + Python).
- Public GitHub repository created and accessible.
- Writing a clear [project plan.](https://github.com/TuNgo1407/ml-agents-group12/blob/develop/docs/group12_MLAI_project_plan.pdf)  


---
## Installation Guide

### Prerequisites
- Git
- Conda
- Unity Hub + Unity Editor 2023.2.12f1


###  Install ML-Agents

1. Clone the repository
```bash
git clone "https://github.com/TuNgo1407/ml-agents-group12.git"
cd ml-agents-group12
```
2. Create Python Virtual Environment
```bash
conda create -n mlagents python=3.10.11 && conda activate mlagents
```
3. Inside this Virtual Environment, install ML-Agents
```bash
python -m pip install mlagents==1.1.0
```
4. For Window user, install Pytorch in this env
```bash
pip3 install torch~=1.13.1 -f https://download.pytorch.org/whl/torch_stable.html
```
5. Install all python packages in `ml-agents-group12/requirements.txt` by this command:
```bash
pip install -r requirements.txt 
```
6. Create a folder within `ml-agents-group12` called `results`


Use [**ML-Agents Installation Guide**](https://github.com/DennisSoemers/ml-agents/blob/fix-numpy-release-21-branch/docs/Installation.md) for more detailed and advanced installation 


## Repository Structure


## 📚 References

- [Dennis Soemers fork](https://github.com/DennisSoemers/ml-agents/tree/fix-numpy-release-21-branch?tab=readme-ov-file)  
- [Unity ML-Agents Documentation](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Readme.md)
