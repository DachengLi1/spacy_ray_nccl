# spacy_ray_collective
A collective implementation to spacy_ray instead of grpc. <br />
2000 update: <br />
        Baseline: 703, 1473 (200 evaluation), score 0.71 <br />
        P2P (Double communication):          1367 （200 evaluation）, score 0.04, gradient too large <br />
        Note: In P2P communication, we actually doubled the communication, because we created two identical PS. <br />
              A better way maybe use allreduce.
              
10000 update: <br />
        Baseline (1 worker): 1330s <br />
                 (2 workers): 3816s <br />
        3x slower, which means that ray rpc is very slow. <br />
