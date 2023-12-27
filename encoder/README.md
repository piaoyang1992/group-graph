
## Requirements
```
python                    3.6.5
pytorch                   1.10.1             
torch-geometric           2.0.3
rdkit                     2018.03.1
```
To install RDKit, please follow the instructions here http://www.rdkit.org/docs/Install.html

* `prepare_group_graph.py` contains codes for construction of group graph.
* `graph_loader` contains codes for building molecular dataset of group graph and comparing group graph
## Training
You can run GIN of group graph in classification tasks
```
class_task.py 
```
You can run GIN of group graph in regression tasks
```
regression_task.py 
```
