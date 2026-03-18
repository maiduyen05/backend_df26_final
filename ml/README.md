# Thư mục `ml/`

Đặt file model PyTorch đã huấn luyện vào đây.

## Tên file mặc định

```
ml/model.pt
```

Có thể thay đổi đường dẫn qua biến môi trường `MODEL_PATH` trong file `.env`.

## Định dạng checkpoint

Backend hỗ trợ 2 cách lưu:

### 1. Lưu state_dict (khuyên dùng)

```python
torch.save(model.state_dict(), "ml/model.pt")
```

Sau đó cập nhật `_real_predict()` trong `app/services/model_service.py`:

```python
self._model = YourTransformerModel(...)
self._model.load_state_dict(torch.load("ml/model.pt"))
self._model.eval()
```

### 2. Lưu toàn model

```python
torch.save(model, "ml/model.pt")
```

```python
self._model = torch.load("ml/model.pt")
self._model.eval()
```

## Input/Output của model

| | Shape | Dtype | Mô tả |
|---|---|---|---|
| Input | `(batch, 66)` | `torch.long` | Token IDs (0–99), 0 = padding |
| Attn output | `(batch, seq_len)` | `torch.float` | Attention weights trung bình các heads |
| Plant signals | `(batch, 2)` | `torch.float` | attr_3, attr_6 normalized 0–1 → × 99 |
| Windows | `(batch, 4)` | `torch.float` | start_month, start_day, end_month, end_day |

## Khi chưa có model

Nếu `ml/model.pt` không tồn tại, backend tự động chạy **MOCK MODE**:
sinh dữ liệu giả với phân phối gần giống thực tế để dev/test frontend.
