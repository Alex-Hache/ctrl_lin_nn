# ctrl_lin_nn

A repository with the code to reproduce the results in the paper entitled "Model Learning with Guaranteed Performance Through Embdedded Matrix submanifolds"

## Required packages : 
+ Pytorch
+ geotorch [github](https://github.com/lezcano/geotorch/tree/master)
+ scipy
+ matlab [api](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)


The results presented in each figure can be reproduced by picking any value between braces as the parameter for the corresponding table or figure.

To reproduce Table 1:
```
python main.py --alpha {0.5, 1, 2, 3}
```

To reproduce Table 2:
```
python main.py --lin hinfnn -g {0.5, 0.75}
python main.py --lin hinfnn -g {0.5, 1, 1.5} -e 3500
```

To reproduce table 3 :
```
python main.py --lin h2nn --mu {1, 0.1, 1e-2, 1e-3}
```

To reproduce table  and figure 2 results : 
```
python main.py -d pendulum -m flnssm --lmi hinf -g 1e-3 -e 10000
python main.py -d pendulum -m flnssm --lin hinfnn -g 1e-3 -e 15000 --lr 5e-3
```

Feel free to contact me : alexandre dot hache at imt-tlantique dot fr
