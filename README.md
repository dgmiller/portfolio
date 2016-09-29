
# Derek Miller: Portfolio

I turn business problems into math problems and solve them with code.

## Introduction

This repository contains some of my work in Applied and Computational Mathematics, summer internships, and side projects.

Some examples:
* image recognition via eigenfaces
* calculating customer lifetime value
* maximizing profit and minimizing cost for customer visits
* finding your seat on an airplane

Libraries used: numpy, scipy, pandas, sklearn, networkx, pyodbc, matplotlib, matplotlib's Basemap, geopy, lifetimes, ...

# Examples

```python
# import statements
import tools as t
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
```

## Recommender System

One way to recommend products to customers is to look at what other similar customers bought.
In this case, we define similarity to be the cosine similarity of two feature vectors.

```python
# dictionary converts customer id to customer name
to_name = t.make_to_name()

R = t.recommender('../data/PTTPARTS.csv')
R.similarity_matrix(transpose=True)
print(to_name['1057']+' is similar to...\n')
for i in R.similar_to(1057)[:15]:
    name = to_name[R.from_index[i[0]]]
    if t.use(name):
        print(name,i[1])
```

    Oregon Freeze Dry Foods is similar to...
    
    ('Aveda Corporation', 0.38346821065500492)
    ('Colgate Palmolive Tech Center', 0.36245699167329504)
    ('Novo Nordisk', 0.32069443135208864)
    ('Allied Old English, Inc.', 0.24309684249639285)
    ('North Carolina State Universit', 0.23108408032649064)
    ('Carolina Foods Inc', 0.21473892259683336)
    ('Quali Tech Inc', 0.21118547093938189)
    ('Labomar d.o.o.', 0.19427937057310141)
    ('The Jerky Shoppe', 0.18108065446288402)
    ('G & G Enterprises Inc', 0.17402004840930574)
    ('Bimbo Bakeries USA', 0.17393181644453315)
    ('Lightlife Foods', 0.16918824204598992)



```python
print R.recommend(1057)[:,0]
```

    ['Pawkit' 'AquaLab Lite' 'AquaLab S4TEV' 'AquaLab Trade-In' 'AquaLab Pre'
     'Pawkit 2' 'AquaLab S3' 'AquaLab 4 Dew' 'AquaLink' 'Refurbished AquaLab']


## Marketing Strategy

Trying to sell a new product to old customers can be thought of as a Multi-Armed Bandit problem. We can solve our dilemma using Thompson Sampling.

```python
t.testMAB(3,[.3,.5,.7],[.3,.5,.7],niters=80)
```


![png](images/output_8_4.png)


## Market Segmentation

One method of customer segmentation relies on feature vectors and graph theory models. 
We can construct a representation of a graph by using the similarity matrix already computed in the recommender object.
Then we can use Markov Clustering to find the groups that are most connected.

```python
R = t.recommender('../data/example.csv')
R.similarity_matrix()
G2 = t.nx.Graph(R.D)
t.nx.draw_networkx(G2)
```


![png](images/output_11_1.png)



```python
cust_clusters = t.markov_cluster_algorithm(R.D,2,2)
G = t.nx.Graph(cust_clusters)
t.nx.draw_networkx(G)
```


![png](images/output_15_0.png)


## Customer Lifetime Value

Marketing to the right customers means we should try to figure out who our most valuable customers are.
The clv object implements a statistical model detailed in *"Counting Your Customers" the Easy Way: An Alternative to the Pareto/NBD Model* by Fader, et al
using the lifetimes module at github.com/CamDavidsonPilon/lifetimes.


```python
C = t.CLV()
C.get_data_from_file('../data/pmg01.csv')
C.fit()
```


```python
print(C.results[:5])
```

           frequency  recency     T  monetary_value  predicted_purchases  \
    id                                                                     
    36413       40.0     78.0  81.0    9.267299e+06            43.297793   
    48379       13.0     22.0  24.0    9.234662e+03            39.999265   
    4017        47.0     93.0  96.0    8.108095e+03            43.498003   
    9946         2.0     32.0  44.0    6.468293e+04             4.656563   
    4885        76.0     94.0  96.0    3.906934e+03            70.049703   
    
           predicted_trans_profit  clv_estimation  prob_alive  
    id                                                         
    36413            9.262273e+06    4.010360e+08    0.997020  
    48379            9.223589e+03    3.689368e+05    0.994689  
    4017             8.105548e+03    3.525752e+05    0.997455  
    9946             6.401612e+04    2.980951e+05    0.978090  
    4885             3.906559e+03    2.736533e+05    0.998260  


## Visualizing Customer Locations

Using the `Geocode` object in `tools.py`, we can convert customer addresses into GPS coordinates.
We can segment our customers by area and find area hubs. The area hubs in this image were calculated with Mean Shift clustering.
Then we can further segment customers by CLV, market, or other features.

![png](images/customers_by_area.png)

## School Projects

*Sobel Filter*


![png](images/cameraman2.png)


*Eigenfaces*


![png](images/meanface.png)


![png](images/minusmean.png)



Also check out a collaborative data visualization of Classic Literature at https://www.behance.net/thesarahkay where I helped with data cleaning.
