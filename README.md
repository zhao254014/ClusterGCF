# ClusterGCF
Cluster-based Graph Collaborative Filtering
## Environment Settings
- Tensorflow-gpu version:  1.11.0

## Example to run the codes.

# gowalla
Run ClusterGCF.py
```
python ClusterGCF.py --dataset gowalla  --regs [1e-4] --embed_size 64 --layer_size [64,64,64,64,64,64] --lr 0.001 --batch_size 2048 --epoch 2000 --groups 3 --temperature 0.1
```
## Reference:

If you have any questions for our paper or codes, please send an email to zhao254014@gmail.com.
