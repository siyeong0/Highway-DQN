# Highway DQN
Reinforcement learning project to train the highway driver using deep q learning algorithm.
## Train
```python
!python train.py --num-ep 20000 
```

![plot](https://github.com/siyeong0/Highway-RL/assets/117014820/150aed14-b72d-46f6-8d7d-f0577e617a30)


## Test
```python
!python test.py --model-path ./checkpoints/highway-fast-v0/highway-fast-v0.pth 
```
<p align="center">
    <img src="https://github.com/siyeong0/Highway-RL/assets/117014820/07941c4d-5343-4dd8-b8cc-c4fb58f2cffb"><br/>
    <em>The highway-fast-v0 environment.</em>
</p>

## Pretreind Model
[Drive link](https://drive.google.com/drive/folders/1pNZsgTemooiZHtzgXY55rG-Hk8YYF5-C?usp=share_link)

## References

* [HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv)
* [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) 
* [Pytorch DQN Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
