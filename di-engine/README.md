# 强化学习

## 月球登陆

训练

```bash
cd ~/examples/di-engine/lunarlander
kubectl create -f train.yaml
```

评估

```bash
kubectl create -f evaluate.yaml
```

部署（演示）

```bash
kubectl create -f deploy.yaml
```

## 超级马里奥兄弟

训练

```bash
cd ~/examples/di-engine/super-mario-bros
kubectl create -f train.yaml
```

部署（演示）

```bash
kubectl create -f deploy.yaml
```

## 史莱姆排球

训练

```bash
cd ~/examples/di-engine/slime-volleyball
kubectl create -f train.yaml
```
