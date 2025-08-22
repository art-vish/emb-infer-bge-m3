import asyncio
import time
from typing import List, Tuple, Any, Callable
from dataclasses import dataclass
from fastapi import HTTPException, status

from src.core.config import BATCH_SIZE, BATCH_TIMEOUT_MS, MAX_QUEUE_SIZE
from src.models.schemas import EmbeddingRequest, BGEEmbeddingResponse

@dataclass
class BatchItem:
    """Элемент батча - запрос и future для результата"""
    request: EmbeddingRequest
    future: asyncio.Future
    timestamp: float
    original_input_length: int  # Для правильного разбора результатов

class BatchProcessor:
    """Процессор батчей для эффективной обработки запросов"""
    
    def __init__(self):
        self.pending_requests: List[BatchItem] = []
        self.batch_lock = asyncio.Lock()
        self.processing_semaphore = asyncio.Semaphore(2)  # Параллельная обработка батчей
        self.stats = {
            "total_batches": 0,
            "total_requests": 0,
            "avg_batch_size": 0,
            "last_batch_time": 0
        }
    
    async def add_request(self, request: EmbeddingRequest, process_func: Callable) -> Any:
        """Добавляет запрос в батч и возвращает результат"""
        # Нормализуем входные данные
        if isinstance(request.input, str):
            inputs = [request.input]
        elif isinstance(request.input, list):
            if not request.input:
                # Пустой запрос - обрабатываем сразу
                return await process_func(request, None)
            elif all(isinstance(item, int) for item in request.input):
                inputs = [" ".join(map(str, request.input))]
            else:
                inputs = request.input
        else:
            inputs = [str(request.input)]
        
        # Создаем элемент батча
        future = asyncio.Future()
        batch_item = BatchItem(
            request=EmbeddingRequest(
                model=request.model,
                input=inputs,
                encoding_format=request.encoding_format,
                return_dense=getattr(request, 'return_dense', True),
                return_sparse=getattr(request, 'return_sparse', True),
                return_colbert=getattr(request, 'return_colbert', True),
                user=request.user
            ),
            future=future,
            timestamp=time.time(),
            original_input_length=len(inputs)
        )
        
        # Добавляем в очередь батчинга
        async with self.batch_lock:
            if len(self.pending_requests) >= MAX_QUEUE_SIZE:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Server is currently handling maximum number of requests. Please try again later."
                )
            
            self.pending_requests.append(batch_item)
            
            # Проверяем, нужно ли запустить батч
            should_process = (
                len(self.pending_requests) >= BATCH_SIZE or
                (self.pending_requests and 
                 (time.time() - self.pending_requests[0].timestamp) * 1000 >= BATCH_TIMEOUT_MS)
            )
            
            if should_process:
                # Извлекаем батч для обработки
                batch_to_process = self.pending_requests[:BATCH_SIZE]
                self.pending_requests = self.pending_requests[BATCH_SIZE:]
                
                # Запускаем обработку батча асинхронно
                asyncio.create_task(self._process_batch(batch_to_process, process_func))
        
        # Ждем результат
        return await future
    
    async def _process_batch(self, batch: List[BatchItem], process_func: Callable):
        """Обрабатывает батч запросов"""
        async with self.processing_semaphore:
            try:
                start_time = time.time()
                
                # Собираем все тексты в один большой запрос
                all_texts = []
                request_boundaries = []  # Границы запросов в общем списке
                
                current_index = 0
                for item in batch:
                    start_idx = current_index
                    all_texts.extend(item.request.input)
                    current_index += len(item.request.input)
                    request_boundaries.append((start_idx, current_index))
                
                print(f"🔄 Processing batch: {len(batch)} requests, {len(all_texts)} texts total")
                
                # Создаем объединенный запрос, сохраняя параметры селективной генерации
                first_request = batch[0].request
                return_dense = getattr(first_request, 'return_dense', True)
                return_sparse = getattr(first_request, 'return_sparse', True)
                return_colbert = getattr(first_request, 'return_colbert', True)
                
                combined_request = EmbeddingRequest(
                    model=first_request.model,
                    input=all_texts,
                    encoding_format=first_request.encoding_format,
                    return_dense=return_dense,
                    return_sparse=return_sparse,
                    return_colbert=return_colbert,
                    user=getattr(first_request, 'user', None)
                )
                
                # Обрабатываем весь батч одним вызовом
                from src.main import app
                batch_result = await process_func(combined_request, app.state.encoder)
                
                # Разбираем результаты по отдельным запросам
                for i, (item, (start_idx, end_idx)) in enumerate(zip(batch, request_boundaries)):
                    try:
                        # Извлекаем данные для этого запроса (BGE формат)
                        request_data = batch_result.data[start_idx:end_idx]
                        # Переиндексируем
                        for j, data_item in enumerate(request_data):
                            data_item.index = j
                        
                        # Определяем запрошенные типы векторов из оригинального запроса
                        orig_req = item.request
                        requested_types = []
                        if getattr(orig_req, 'return_dense', True):
                            requested_types.append("dense")
                        if getattr(orig_req, 'return_sparse', True):
                            requested_types.append("sparse")
                        if getattr(orig_req, 'return_colbert', True):
                            requested_types.append("colbert")
                        
                        individual_result = BGEEmbeddingResponse(
                            data=request_data,
                            model=batch_result.model,
                            usage={
                                "prompt_tokens": int(sum(len(text.split()) * 1.3 for text in item.request.input)),
                                "total_tokens": int(sum(len(text.split()) * 1.3 for text in item.request.input))
                            },
                            embedding_types=requested_types
                        )

                        
                        item.future.set_result(individual_result)
                    except Exception as e:
                        print(f"Error processing individual result {i}: {e}")
                        item.future.set_exception(e)
                
                # Обновляем статистику
                batch_time = time.time() - start_time
                self.stats["total_batches"] += 1
                self.stats["total_requests"] += len(batch)
                self.stats["avg_batch_size"] = self.stats["total_requests"] / self.stats["total_batches"]
                self.stats["last_batch_time"] = batch_time
                
                print(f"✅ Batch processed: {len(batch)} requests in {batch_time:.2f}s")
                
            except Exception as e:
                print(f"Error processing batch: {e}")
                # Уведомляем все запросы об ошибке
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(e)
    
    async def start_timeout_processor(self):
        """Запускает процессор таймаутов для обработки неполных батчей"""
        while True:
            try:
                await asyncio.sleep(BATCH_TIMEOUT_MS / 1000)  # Конвертируем в секунды
                
                async with self.batch_lock:
                    if (self.pending_requests and 
                        (time.time() - self.pending_requests[0].timestamp) * 1000 >= BATCH_TIMEOUT_MS):
                        
                        # Извлекаем все ожидающие запросы
                        batch_to_process = self.pending_requests[:]
                        self.pending_requests.clear()
                        
                        print(f"⏰ Timeout batch: {len(batch_to_process)} requests")
                        
                        # Определяем функцию обработки по типу первого запроса
                        # Используем BGE по умолчанию для таймаут батчей
                        from src.services.model_service import process_bge_embeddings
                        asyncio.create_task(self._process_batch(batch_to_process, process_bge_embeddings))
                        
            except Exception as e:
                print(f"Error in timeout processor: {e}")
                await asyncio.sleep(1)

# Глобальный экземпляр процессора батчей
batch_processor = BatchProcessor()

async def enqueue_batch_request(request: EmbeddingRequest, process_func: Callable) -> Any:
    """Добавляет запрос в батч для обработки"""
    return await batch_processor.add_request(request, process_func)

def get_batch_stats():
    """Возвращает статистику батчинга"""
    return batch_processor.stats
