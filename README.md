# 动态数字键盘：Flask后端

### 维护者

- 陈想东

### 运行

```
python app.py
```

### 访问地址
```
localhost:5000
```

### APIs 可访问接口
```
GET: localhost:5000/problems 获取离散数学题目列表
```
```
POST: localhost:5000/api/predict 获取算法产生的符号
param: question
如：
axios.post('http://localhost:5000/api/predict', question)
```
