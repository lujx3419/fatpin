# GitHub 上传指南

## 准备工作

### 1. 设置 Git 用户信息（如果还没有设置）

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

或者仅为这个仓库设置：
```bash
cd /data/satori_hdd1/lujianxu/reinforce/fatpin
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

### 2. 提交代码到本地仓库

```bash
cd /data/satori_hdd1/lujianxu/reinforce/fatpin
git add .
git commit -m "Initial commit: FatPIN reinforcement learning code with PortPy support"
```

或者运行自动脚本：
```bash
./setup_github.sh
```

## 创建 GitHub 仓库

### 方法 1: 通过 GitHub 网页

1. 访问 https://github.com/new
2. 输入仓库名称（例如：`fatpin`）
3. 选择 Public 或 Private
4. **不要**勾选 "Initialize this repository with a README"（因为我们已经有了 README）
5. 点击 "Create repository"

### 方法 2: 使用 GitHub CLI（如果已安装）

```bash
gh repo create fatpin --public --source=. --remote=origin --push
```

## 推送代码到 GitHub

### 使用 HTTPS（推荐首次使用）

```bash
git remote add origin https://github.com/YOUR_USERNAME/fatpin.git
git branch -M main
git push -u origin main
```

### 使用 SSH（如果已配置 SSH 密钥）

```bash
git remote add origin git@github.com:YOUR_USERNAME/fatpin.git
git branch -M main
git push -u origin main
```

## 验证上传

上传成功后，访问你的 GitHub 仓库页面：
```
https://github.com/YOUR_USERNAME/fatpin
```

你应该能看到所有的代码文件。

## 后续更新

如果要更新代码：

```bash
git add .
git commit -m "Update: 描述你的更改"
git push
```

## 注意事项

- `.gitignore` 文件已经配置好，会排除 `__pycache__`、`result/`、`.pth` 模型文件等
- 如果 `portpy_data` 目录包含大型数据文件，可能需要使用 Git LFS：
  ```bash
  git lfs install
  git lfs track "*.h5"
  git add .gitattributes
  ```

