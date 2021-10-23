# MSCL
Code for Multisample-based Contrastive Loss for Top-k Recommendation  (IEEE TMM).

What we propose is a loss function that can be applied to many methods. Our code is based on the LightGCN. Thanks for the previous work and code https://github.com/gusye1234/LightGCN-PyTorch . We followed its setup, which you can refer to.


Use Cpp Extension in  `code/sources/`  for negative sampling. To use the extension, please install `pybind11` and `cppimport` under your environment.


The parameter adjustment is relatively simple, and the following is available for reference.

python main.py   --layer 3   --dataset="amazon-book"   --temperature 0.1            --info sgk15_alpha0.45 

python main.py   --layer 1  --dataset="ifashion"    --decay 1e-5  --temperature 0.2    --info sgk15_alpha0.60 

python main.py   --layer 2  --dataset="yelp2018"      --temperature 0.2              --info sgk15_alpha0.45 
