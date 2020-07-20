# CKAN

This repository is the implementation of CKAN :

> CKAN: Collaborative Knowledge-aware Attentive Network for Recommender Systems，SIGIR 2020
>
> Ze Wang, Guangyan Lin, Huobin Tan, Qinghong Chen, and Xiyang Liu

![](https://github.com/weberrr/CKAN/blob/master/framework.png)

## Required packages

The code has been tested running under Python 3.6.9, with the following packages installed (along with their dependencies):
- torch==1.3.0
- torchvision==0.4.1
- numpy==1.17.3
- scikit-learn==0.21.3

## Files in the folder

- `data/`
  - `music/` 
    - `user_artists.dat`: raw rating file of Last.FM dataset;
    - `kg.txt`: knowledge graph file;
    - `item_index2entity_id.txt`: the mapping from item indices in the raw rating file to entity IDs in the KG;
  - `book/` ( the structure of other datasets is similar )
  - `movie/`
  - `restaurant/`
- `src/`: implementations of CKAN.

## Perpare  & preprocess data

We have prepared processed data in `music` and `book` . You can skip this step and proceed directly to the next step. But for larger dataset ( `movie` and `restaurant` ), you need to download and preprocess yourself by following the steps below:


- Music

```
$ wget http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip
$ unzip hetrec2011-lastfm-2k.zip
$ mv hetrec2011-lastfm-2k/user_artists.dat data/music/
$ cd src
$ python preprocess.py --dataset music
$ python main.py
```


- Book
```
$ wget http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip
$ unzip BX-CSV-Dump.zip
$ mv BX-CSV-Dump/BX-Book-Ratings.csv data/book/
$ cd src
$ python preprocess.py --dataset book
```

- movie
```
$ wget http://files.grouplens.org/datasets/movielens/ml-20m.zip
$ unzip ml-20m.zip
$ mv ml-20m/ratings.csv data/movie/
$ cd src
$ python preprocess.py --dataset movie
```

- Restaurant
```
$ wget https://github.com/hwwang55/KGNN-LS/raw/master/data/restaurant/Dianping-Food.zip
$ unzip Dianping-Food.zip
$ mv Dianping-Food/ data/restaurant
```


##  Run the code

We set a random seed to facilitate users to observe the effect of the model easily. You can reset the random seed by adding parameters this way:  `--random_flag True`

- music

```
$ cd src
$ python main.py --dataset music (note: use -h to check optional arguments)
```

- book 

```
$ cd src
$ python main.py --dataset book --n_layer 2 --user_triple_set_size 16
```

- movie

```
$ cd src
$ python main.py --dataset movie --n_layer 1 --user_triple_set_size 32
```

- restaurant

```
$ cd src
$ python main.py --dataset restaurant --n_layer 1 --user_triple_set_size 16
```

