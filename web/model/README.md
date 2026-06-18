# Web Model Assets

Setelah training dan export ONNX di Kaggle selesai, salin isi folder:

```text
/kaggle/working/outputs/web_model
```

ke folder ini:

```text
web/model
```

Minimal file yang dibutuhkan:

```text
model.onnx
vocab.txt
config.json
label_map.json
```

Jika tersedia, salin juga:

```text
tokenizer.json
tokenizer_config.json
special_tokens_map.json
```

`model.onnx` tidak disimpan di Git jika ukurannya besar.
