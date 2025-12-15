# Kubernetes Deployment

## Prerequisites

- Kubernetes cluster with NVIDIA GPU support
- nvidia-device-plugin installed
- Node with A2000 GPU (or adjust nodeSelector in deployment.yaml)
- kubectl configured to access your cluster

## Quick Start

### 1. Create namespace

```bash
kubectl apply -f namespace.yaml
```

### 2. Create secret (ВАЖНО: сначала отредактируйте!)

Отредактируйте `secret.yaml` и замените `CHANGE_THIS_TO_YOUR_SECURE_TOKEN` на реальный токен:

```bash
# Генерация случайного токена
openssl rand -base64 32

# Применение секрета
kubectl apply -f secret.yaml
```

### 3. Create PVC

```bash
kubectl apply -f pvc.yaml
```

### 4. Deploy application

```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

### 5. Check status

```bash
# Проверка статуса пода
kubectl -n ml-services get pods

# Просмотр логов
kubectl -n ml-services logs -f deployment/emb-infer-bge-m3

# Проверка сервиса
kubectl -n ml-services get svc
```

## Verify GPU allocation

```bash
kubectl -n ml-services exec deployment/emb-infer-bge-m3 -- nvidia-smi
```

## Configuration

### Параметры для A2000 (6GB VRAM)

В `deployment.yaml` уже установлены оптимальные значения для A2000:

- `BATCH_SIZE=8` - небольшой батч для экономии памяти
- `PROCESSING_CONCURRENCY=2` - не больше, чтобы не словить OOM
- `MAX_QUEUE_SIZE=50` - максимум запросов в очереди

Если у вас другая видеокарта, скорректируйте эти значения и `nodeSelector`.

## Troubleshooting

### Pod не запускается

```bash
# Проверьте события
kubectl -n ml-services describe pod -l app=emb-infer-bge-m3

# Проверьте, что секрет создан
kubectl -n ml-services get secret emb-infer-secrets
```

### GPU не выделяется

```bash
# Проверьте, что nvidia-device-plugin работает
kubectl get pods -n kube-system | grep nvidia

# Проверьте ноды с GPU
kubectl get nodes -o=custom-columns=NAME:.metadata.name,GPU:.status.allocatable."nvidia\.com/gpu"
```

### OOM ошибки

Уменьшите `BATCH_SIZE` и/или `PROCESSING_CONCURRENCY` в `deployment.yaml`.

## Clean up

```bash
kubectl delete -f service.yaml
kubectl delete -f deployment.yaml
kubectl delete -f pvc.yaml
kubectl delete -f secret.yaml
kubectl delete -f namespace.yaml
```

## Production Notes

- **Не коммитьте** `secret.yaml` с реальным токеном в git!
- Используйте Sealed Secrets или Vault для production секретов
- Настройте мониторинг (Prometheus metrics доступны на `/metrics`)
- Настройте backups для PVC с моделью

