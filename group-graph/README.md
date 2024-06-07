

## Requirements
```
python                    3.9.13
torch                     1.13.1             
torch-cluster             1.6.1           
torch-scatter             2.1.0           
torch-sparse              0.6.16          
torch_geometric           2.5.2                     
torchmetrics              1.3.2                   
rdkit                     2023.9.5
```
To install RDKit, please follow the instructions here http://www.rdkit.org/docs/Install.html

* `prepare_group_graph.py` 
contains codes for construction of group graph. 
group graphs are save as group_graph.pt  vocabulary are saved as vocab.csv
                        
* `data_loader` contains codes for building molecular dataset of group graph

## Training
You can run GIN of group graph in classification tasks for molecular property prediction
```
class_task.py 
```
You can run GIN of group graph in regression tasks  for molecular property prediction
```
regression_task.py 
```
You can run GIN of group graph in classification tasks for drug-drug interaction prediction
```
ddi_predict.py 
```