# spacy_ray_collective
This github is inherited from spacy-ray (dev branch). <br />
<br />
It trains model using the new collective calls available in current ray-1.2-dev, replacing the original get(), set() usage in spacy-ray. <br />
<br />
The runtime comparison for 1000 update using spacy pipeline = ["tok2vec", "ner"] is showed below: <br />

    | Comparison    | spacy-ray     | spacy-ray-nccl |  ratio  |  
    | ------------- | ------------- | -------------- | ------- | 
    | 1 worker      | 137.5 ± 2.1   | 116.7 ± 2.51   |  1.18x  |
    | 2 workers     | 354.1 ± 16.8  | 171.1 ± 1.11   |  2.07x  |  
    | 4 workers     | 523.9 ± 10.4  | 179.6 ± 2.91   |  2.92x  |  
    | 8 workers     | 710.1 ± 3.0   | 205.8 ± 1.20   |  3.45x  | 
    | 16 workers    | 1296.1 ± 42.1 | 248.3 ± 3.63   |  5.22x  | 

Mean and standard deviation are obtained by three trials (unit: second). Each worker is on a different node.  <br />
<br />
#### Runtime comparison: <br />

The ideal plot should be a horizontal line. <br />
<br />
![runtime](results/time_comparison.PNG) <br />
<br />

#### Speedup comparison: <br /> 
The trivial speedup is a horizontal line y = 1. <br />
 <br />
![speedup](results/ratio_comparison.PNG) <br />
    
 <br />
 
 #### Scalability comparison: <br /> 
The ideal scalability is the line y = x. <br />
 <br />
![sscalability](results/scalability_comparison.PNG) <br />
    
 <br />
 
 ### To install the necessary module: <br />
 
1. ```conda create -n spacy-ray python=3.7.3``` <br />
2. ```conda activate spacy-ray``` <br />
3. ```pip install spacy-nightly[cuda]``` <br />
 &nbsp;    - This will take some time, if observe a build error in cupy, try: ```pip install cupy-cuda[version]``` <br />
 &nbsp;      e.g. for cudatoolkit 11.0: ```pip install cupy-cuda110``` <br />
4. ```pip install spacy-ray``` <br />
 &nbsp;    - run ```python -m spacy ray --help``` to check whether install correctly <br />
5. The collective calls are only available in current ray github. Instead we use the latest ray-1.1 in pip to test runtime. <br />
 &nbsp;    - get collective code:     ```git clone https://github.com/ray-project/ray``` <br />
 &nbsp;    - access the installed code of ray 1.1:    ```cd [path-to-packages]/ray``` <br />
 &nbsp;     If using conda, typically the path would be ```[path-to-conda]/anaconda3/envs/spacy-ray/lib/python3.7/site-packages/``` <br />
 &nbsp;    - copy the code over: ```cp -r [path-to-github-ray]ray/python/ray/util/collective ./ray/util``` <br />
 &nbsp;    - add to "init" file: ```vim ./ray/util/__init__.py``` -> ```from ray.util import ray```, append "collective" to the "all" dict. <br />
6. The last step is to replace the installed spacy-ray using this github. <br />
 &nbsp;   - ```git clone https://github.com/MccRee177/spacy_ray_nccl``` <br />
 &nbsp;   - ```mv [path-to-github-spacy-ray-nccl] [path-to-packages]``` <br />
 &nbsp;   - make a copy of the original spacy_ray in case you would like to recover the comparison:  <br \ >
 &nbsp:     ```mv [path-to-packages]/spacy_ray [path-to-packages]/spacy_ray_original``` <br />
 &nbsp;   - ```mv [path-to-packages]/spacy_ray_nccl [path-to-packages]/spacy_ray``` <br />

### To run examples: <br />

1. ```git clone https://github.com/MccRee177/spacy_ray_example``` <br />
2. ```cd spacy_ray_example/tmp/experiments/en-ent-wiki``` <br />
3. Setup the ray cluster in different machine. The code will detect the available ray cluster and attach. <br />
4. Modify the config (for training hyperparameter) and project.yml (for number of workers) <br />
5. Download and process necessary files: (reference: https://github.com/explosion/spacy-ray/tree/develop/tmp/experiments/en-ent-wiki) <br />
 &nbsp;    - ```spacy project assets``` <br />
 &nbsp;    - ```spacy project run corpus``` <br />
6. ```spacy project run ray-train``` <br />

### Evaluation note: <br />

The github turns off the score evaluation for comparison. This is because evaluation takes a long time, and we only want to measure the speedup during training. <br />
<br />
To turn on: comment the hard-coded socres in train() function at worker.py, change the if condition from if self.rank ==0 to if True, and uncomment socres = self.evaluate() <br />

### Implementation note: <br />

The original spacy-ray uses sharded parameter server and update the model parameter in each worker asynchronizely. The current implementation uses all_reduce strategy, which performs similarly to a sharded parameter server. It has a whole copy of the model parameter in each worker. For update, it uses collective.allreduce() to synchronize gradients. <br />

### Ray cluster setup node:  <br />

A template for setting up a 16 machine ray cluster: <br />
```
  1 #!/bin/bash 
  2 
  3 MY_IPADDR=$(hostname -i) 
  4 echo $MY_IPADDR 
  5 
  6 ray stop --force 
  7 sleep 3 
  8 ray start --head --port=6380 --object-manager-port=8076  --object-store-memory=32359738368 
  9 sleep 2 
 10 
 11 for i in {1..15} 
 12 do 
 13   echo "=> node $i" 
 14   ssh -o StrictHostKeyChecking=no h$i.ray-dev-16.BigLearning "cd spacy_ray_example/tmp/experiments/en-ent-wiki;  source ~/anaconda3/bin/activate; conda activate spacy-ray; ray stop --force; ray start --address='$MY_IPADDR:6380' --object-manager-port=8076 --object-store-memory=32359738368"; 
 15 done 
 16 wait 
```
    
