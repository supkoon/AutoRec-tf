# AutoRec-tf



## --reference paper

Suvash Sedhain, Aditya Krishna Menon, Scott Sanner, and Lexing Xie. 2015. AutoRec: Autoencoders Meet Collaborative Filtering. In Proceedings of the 24th International Conference on World Wide Web (WWW '15 Companion). Association for Computing Machinery, New York, NY, USA, 111–112. DOI:https://doi.org/10.1145/2740908.2742726

## --discription
dataset : Movielens-1m  
loss : masked MSE loss


## example : I-AutoRec
```
python fm.py --path "./datasets/ml-1m" --dataset "ratings.dat" --kind I --layers [300,100,300] --reg 0.01 --learner "SGD" --test_size 0.1 --epochs 10 --batch_size 32 --lr 0.01 --patience 10 --out 1 

```
## example : U-AutoRec

```
python fm.py --path "./datasets/ml-1m" --dataset "ratings.dat" --kind U --layers [300,100,300] --reg 0.01 --learner "SGD" --test_size 0.1 --epochs 10 --batch_size 32 --lr 0.01 --patience 10 --out 1 
```
