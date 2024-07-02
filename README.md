# Analysis of Demands in Bias by Model Itself: SVD-based Dual Contradictory Learning Framework

The official implementation for the paper titled: "Analysis of Demands in Bias by Model Itself: SVD-based Dual Contradictory Learning Framework"

Version of Python: `3.7.11`

Versions of dependent package are in `requirement.txt`

Setting break point: `./src/configs/model/*.yaml`

Run the code: 

``` shell
cd ./src
python main.py -m {model_name} -d {dataset}
```

Such as:

``` shell
cd ./src
python main.py -m LayerGCN -d clothing
```

The default option of model is `SELFCFED_LGN`

The default option of dataset is `baby`
